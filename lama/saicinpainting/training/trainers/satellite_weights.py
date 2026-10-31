"""Explicit generator warm start; does not resume optimizer/trainer state."""
import logging

import torch

LOGGER = logging.getLogger(__name__)


def adapt_satellite_generator_weights(generator, weights, allow_expansion=False):
    target = generator.state_dict()
    first_conv = next((name for name, module in generator.named_modules()
                       if isinstance(module, torch.nn.Conv2d)), None)
    first_weight = (first_conv + '.weight' if first_conv else 'weight') if first_conv is not None else None
    result = dict(weights)
    for name, value in target.items():
        if name not in result or result[name].shape == value.shape:
            continue
        old = result[name]
        can_expand = (allow_expansion and name == first_weight and value.ndim == 4
                      and old.ndim == 4 and value.shape[1] == 5 and old.shape[1] == 4
                      and old.shape[0] == value.shape[0] and old.shape[2:] == value.shape[2:])
        if not can_expand:
            raise ValueError('Incompatible pretrained generator tensor {}: {} -> {}'.format(
                name, tuple(old.shape), tuple(value.shape)))
        expanded = value.new_zeros(value.shape)
        expanded[:, :4] = old.to(device=value.device, dtype=value.dtype)
        result[name] = expanded
        LOGGER.info('Expanded %s from 4 to 5 channels; satellite weights initialized to zero', name)
    return result


def load_pretrained_generator(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state = checkpoint.get('state_dict', checkpoint)
    weights = {name[len('generator.'):]: value for name, value in state.items()
               if name.startswith('generator.')}
    if not weights:
        raise ValueError('Expected a LaMa checkpoint containing generator.* tensors')
    weights = adapt_satellite_generator_weights(
        model.generator, weights, allow_expansion=getattr(model, 'condition_on_satellite', False))
    model.generator.load_state_dict(weights, strict=True)
    LOGGER.info('Loaded generator from %s; discriminator and optimizer start fresh', checkpoint_path)
