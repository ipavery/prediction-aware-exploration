#!/usr/bin/env python3
"""Single-GPU scratch LaMa training on manifest-indexed PNG observations."""
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
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
import torch
from torch.utils.data import DataLoader
import wandb

from saicinpainting.png_dataset import PngMapDataset, unknown_metrics, scores, save_preview
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
        total, metrics = super().generator_loss(batch)
        # Retain LaMa GAN/feature matching, but supervise only original-map pixels in L1.
        mask, valid = batch['mask_for_losses'], batch['valid_region']
        weights = mask * self.config.losses.l1.weight_missing + (1 - mask) * self.config.losses.l1.weight_known
        error = torch.abs(batch[self.image_to_discriminator] - batch['image'])
        l1 = (error * weights * valid).sum() / (valid.sum() * error.shape[1]).clamp_min(1)
        total = total - metrics['gen_l1'] + l1
        metrics['gen_l1'] = l1
        return total, metrics

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        result = self(batch)
        prediction = result['predicted_image'].detach().float().mean(1).cpu().numpy()
        target = batch['image'][:, 0].detach().float().cpu().numpy()
        mask = batch['mask'][:, 0].detach().cpu().numpy()
        counts = unknown_metrics(prediction, target, mask)
        if batch_idx < self.config.png_data.preview_count and not self.sanity_checking():
            sample = {k: v[0].detach().float().cpu().numpy() for k, v in batch.items()
                      if k in ('image', 'observation', 'mask', 'satellite')}
            save_preview(sample, prediction[0], Path('png_samples') / f'epoch_{self.current_epoch:03d}' / f'val_{batch_idx:03d}.png')
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
            record = dict(epoch=self.current_epoch, global_step=self.global_step, counts=pooled, **values)
            with open('validation.jsonl', 'a') as stream:
                stream.write(json.dumps(record) + '\n')
        LOG.info('Unknown-region validation: %s', values)


@hydra.main(config_path='../configs/training', config_name='png-maps')
def main(config):
    if config.get('pretrained_generator_path'):
        raise ValueError('Scratch experiment cannot load pretrained_generator_path; use resume_from_checkpoint for continuation')
    pl.seed_everything(config.seed)
    # Existing training visualization calls wandb.log; a disabled run keeps it local and network-free.
    wandb.init(mode='disabled', project='lama-png-local')
    config.visualizer.outdir = str(Path(config.visualizer.outdir).resolve())
    OmegaConf.save(config, 'config.yaml')
    csv_logger = CSVLogger('metrics', name='csv')
    tb_logger = TensorBoardLogger(config.location.tb_dir, name=Path.cwd().name)
    model = PngTrainingModule(config)
    LOG.info('Training %d PNG pairs; monitoring %d validation pairs', len(model.train_set), len(model.val_set))
    callback = ModelCheckpoint(dirpath='models', monitor='val_unknown_iou', mode='max',
        save_top_k=3, save_last=True, filename='{epoch:03d}-{val_unknown_iou:.4f}')
    trainer = pl.Trainer(callbacks=[callback], logger=[csv_logger, tb_logger],
                         default_root_dir=str(Path.cwd()), **OmegaConf.to_container(config.trainer.kwargs, resolve=True))
    trainer.fit(model)
    Path('best_checkpoint.txt').write_text(callback.best_model_path + '\n')
    Path('last_checkpoint.txt').write_text(callback.last_model_path + '\n')
    wandb.finish()


if __name__ == '__main__':
    main()
