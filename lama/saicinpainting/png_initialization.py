"""Initialize LaMa weights without restoring the source training run."""
from collections.abc import Mapping
import hashlib
from pathlib import Path


def checkpoint_components(checkpoint, discriminator='auto'):
    mode = str(discriminator).lower()
    if mode not in ('auto', 'true', 'false'):
        raise ValueError('pretrained_discriminator must be auto, true, or false')
    if not isinstance(checkpoint, Mapping):
        raise ValueError('Expected a LaMa .ckpt dictionary, not a TorchScript .pt model')
    state = checkpoint.get('state_dict', checkpoint)
    if not isinstance(state, Mapping):
        raise ValueError('Checkpoint state_dict must be a mapping')
    generator = {k[len('generator.'):]: v for k, v in state.items()
                 if isinstance(k, str) and k.startswith('generator.')}
    available_discriminator = {k[len('discriminator.'):]: v for k, v in state.items()
                               if isinstance(k, str) and k.startswith('discriminator.')}
    if not generator:
        raise ValueError('Expected generator.* tensors in a LaMa training .ckpt')
    if mode == 'true' and not available_discriminator:
        raise ValueError('Discriminator initialization was required, but checkpoint has no discriminator.* tensors')
    return generator, available_discriminator if mode != 'false' else {}


def initialize_from_checkpoint(model, checkpoint_path, discriminator='auto'):
    import torch
    from saicinpainting.training.trainers.satellite_weights import adapt_satellite_generator_weights

    source = Path(checkpoint_path).expanduser().resolve()
    digest = hashlib.sha256()
    with source.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    checkpoint = torch.load(str(source), map_location='cpu')
    generator, discriminator_weights = checkpoint_components(checkpoint, discriminator)
    generator = adapt_satellite_generator_weights(model.generator, generator,
        allow_expansion=bool(getattr(model, 'condition_on_satellite', False)))
    model.generator.load_state_dict(generator, strict=True)
    if discriminator_weights:
        model.discriminator.load_state_dict(discriminator_weights, strict=True)
    return dict(kind='base_checkpoint', source=str(source), sha256=digest.hexdigest(),
                generator_initialized=True, discriminator_initialized=bool(discriminator_weights),
                satellite_channel_zero_initialized=bool(getattr(model, 'condition_on_satellite', False)),
                source_epoch=checkpoint.get('epoch'), source_global_step=checkpoint.get('global_step'),
                optimizer_state_restored=False, trainer_state_restored=False)
