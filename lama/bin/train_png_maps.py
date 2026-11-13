#!/usr/bin/env python3
"""Single-GPU LaMa training on PNG observations, with optional base weights."""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ.setdefault('WANDB_MODE', 'disabled')

import json
import logging
from pathlib import Path

import hydra
import numpy as np
from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger, WandbLogger
import torch
from torch.utils.data import DataLoader
import wandb

from saicinpainting.png_dataset import PngMapDataset, unknown_metrics, scores, save_preview
from saicinpainting.png_training_masks import loss_inputs
from saicinpainting.png_initialization import initialize_from_checkpoint
from saicinpainting.training.trainers.default import DefaultInpaintingTrainingModule

LOG = logging.getLogger(__name__)


class PngTrainingModule(DefaultInpaintingTrainingModule):
    def __init__(self, config):
        model_kwargs = dict(config.training_model)
        model_kwargs.pop('kind')
        super().__init__(config, use_ddp=False, **model_kwargs)
        options = config.png_data
        shared = dict(root=options.root, use_satellite=config.training_model.condition_on_satellite,
                      crop_size=options.crop_size, padding_max=options.padding_max, seed=config.seed)
        self.train_set = PngMapDataset(options.train_manifest, training=True,
            satellite_shift_px=options.satellite_shift_px, satellite_dropout=options.satellite_dropout, **shared)
        self.val_set = PngMapDataset(options.val_manifest, training=False, limit=options.val_limit, **shared)
        self.shuffle_generator = torch.Generator().manual_seed(config.seed)

    def sanity_checking(self):
        # The repository pins Lightning 1.2.9, which calls this running_sanity_check.
        return getattr(self.trainer, 'sanity_checking', getattr(self.trainer, 'running_sanity_check', False))

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.config.data.batch_size, shuffle=True,
                          num_workers=self.config.data.num_workers, pin_memory=True,
                          generator=self.shuffle_generator, persistent_workers=False)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=1, shuffle=False,
                          num_workers=self.config.data.num_workers, pin_memory=True)

    def on_train_epoch_start(self):
        self.train_set.set_epoch(self.current_epoch)
        # Makes shuffle order resumable without depending on previous model RNG calls.
        self.shuffle_generator.manual_seed(self.config.seed + self.current_epoch)

    def forward(self, batch):
        target = batch['image']
        batch['image'] = batch.get('observation', target)
        try:
            result = super().forward(batch)
        finally:
            batch['image'] = target
        result['inpainted'] = (result['mask'] * result['predicted_image'] +
                              (1 - result['mask']) * result.get('observation', target))
        return result

    def generator_loss(self, batch):
        supervised = loss_inputs(batch, self.image_to_discriminator)
        total, metrics = super().generator_loss(supervised)
        # Normalize reconstruction by valid map area rather than the padded canvas.
        mask, valid = supervised['mask_for_losses'], batch['valid_region']
        weights = mask * self.config.losses.l1.weight_missing + (1 - mask) * self.config.losses.l1.weight_known
        error = torch.abs(supervised[self.image_to_discriminator] - batch['image'])
        l1 = (error * weights * valid).sum() / (valid.sum() * error.shape[1]).clamp_min(1)
        total = total - metrics['gen_l1'] + l1
        metrics['gen_l1'] = l1
        return total, metrics

    def discriminator_loss(self, batch):
        return super().discriminator_loss(loss_inputs(batch, self.image_to_discriminator))

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        result = self(batch)
        prediction = result['predicted_image'].detach().float().mean(1).cpu().numpy()
        target = batch['image'][:, 0].detach().float().cpu().numpy()
        mask = batch['evaluation_mask'][:, 0].detach().cpu().numpy()
        counts = unknown_metrics(prediction, target, mask)
        if batch_idx < self.config.png_data.preview_count and not self.sanity_checking():
            sample = {k: v[0].detach().float().cpu().numpy() for k, v in batch.items()
                      if k in ('image', 'observation', 'mask', 'satellite', 'valid_region', 'evaluation_mask')}
            preview = Path('png_samples') / f'epoch_{self.current_epoch:03d}' / f'val_{batch_idx:03d}.png'
            save_preview(sample, prediction[0], preview)
            if wandb.run is not None and os.environ.get('WANDB_MODE', 'disabled') != 'disabled':
                wandb.log({f'validation/case_{batch_idx:02d}': wandb.Image(str(preview),
                           caption=f'Epoch {self.current_epoch}, validation case {batch_idx}'),
                           'trainer/global_step': self.global_step}, commit=False)
        return counts

    def validation_epoch_end(self, outputs):
        if not outputs:
            raise RuntimeError('Validation returned no samples')
        pooled = {k: sum(row[k] for row in outputs) for k in outputs[0]}
        values = scores(pooled)
        # Undefined precision/IoU is logged as zero; raw counts preserve the distinction.
        metrics = {'val_unknown_iou': values['occupied_iou'] or 0.,
                   'val_unknown_f1': values['occupied_f1'] or 0.,
                   'val_unknown_mae': values['unknown_mae'] or 0.}
        self.log_dict(metrics, prog_bar=True, logger=True)
        if not self.sanity_checking():
            record = dict(epoch=self.current_epoch, global_step=self.global_step,
                          padding_policy=self.config.png_data.padding_policy, counts=pooled, **values)
            with open('validation.jsonl', 'a') as stream:
                stream.write(json.dumps(record) + '\n')
        LOG.info('Unknown-region validation: %s', values)


@hydra.main(config_path='../configs/training', config_name='png-maps')
def main(config):
    base_path = config.get('pretrained_generator_path')
    if base_path and config.trainer.kwargs.get('resume_from_checkpoint'):
        raise ValueError('Choose base initialization OR resume_from_checkpoint, not both')
    pl.seed_everything(config.seed)
    config.visualizer.outdir = str(Path(config.visualizer.outdir).resolve())
    OmegaConf.save(config, 'config.yaml')
    csv_logger = CSVLogger('metrics', name='csv')
    tb_logger = TensorBoardLogger(config.location.tb_dir, name=Path.cwd().name)
    loggers = [csv_logger, tb_logger]
    mode = os.environ.get('WANDB_MODE', 'disabled').strip().lower()
    if mode not in ('disabled', 'offline', 'online'):
        raise ValueError('WANDB_MODE must be disabled, offline, or online')
    os.environ['WANDB_MODE'] = mode
    if mode == 'disabled':
        # The inherited training previews call wandb.log even when remote tracking is off.
        wandb.init(mode='disabled', project=os.environ.get('WANDB_PROJECT', 'lama-png'))
    else:
        # API credentials are read by W&B from its environment/login, never put in config.yaml.
        wb_logger = WandbLogger(name=Path.cwd().name,
            project=os.environ.get('WANDB_PROJECT', 'lama-png'),
            entity=os.environ.get('WANDB_ENTITY') or None,
            group=os.environ.get('WANDB_RUN_GROUP') or None,
            save_dir=str(Path.cwd()), offline=(mode == 'offline'), log_model=False,
            config=OmegaConf.to_container(config, resolve=True))
        run = wb_logger.experiment  # Initialize once before inherited wandb.log calls.
        loggers.append(wb_logger)
        LOG.info('W&B tracking mode: %s; run ID: %s', mode, run.id)
    model = PngTrainingModule(config)
    if base_path:
        initialization = initialize_from_checkpoint(model, base_path,
            discriminator=config.get('pretrained_discriminator', 'auto'))
        LOG.info('Initialized base generator; discriminator initialized: %s; source SHA256: %s',
                 initialization['discriminator_initialized'], initialization['sha256'])
    elif config.trainer.kwargs.get('resume_from_checkpoint'):
        initialization = dict(kind='resume', checkpoint=str(config.trainer.kwargs.resume_from_checkpoint))
    else:
        initialization = dict(kind='scratch', generator_initialized=False, discriminator_initialized=False)
    Path('initialization.json').write_text(json.dumps(initialization, indent=2, default=str) + '\n')
    if mode != 'disabled':
        wandb.config.update({'initialization': initialization}, allow_val_change=True)
    LOG.info('Training %d PNG pairs; monitoring %d validation pairs', len(model.train_set), len(model.val_set))
    callback = ModelCheckpoint(dirpath='models', monitor='val_unknown_iou', mode='max',
        save_top_k=3, save_last=True, filename='{epoch:03d}-{val_unknown_iou:.4f}')
    trainer = pl.Trainer(callbacks=[callback], logger=loggers,
                         default_root_dir=str(Path.cwd()), **OmegaConf.to_container(config.trainer.kwargs, resolve=True))
    trainer.fit(model)
    Path('best_checkpoint.txt').write_text(callback.best_model_path + '\n')
    Path('last_checkpoint.txt').write_text(callback.last_model_path + '\n')
    wandb.finish()


if __name__ == '__main__':
    main()
