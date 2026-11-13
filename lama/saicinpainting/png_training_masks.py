"""Separate unknown model inputs from valid supervised targets.

These operations work with both Torch tensors and NumPy arrays.
"""


def loss_inputs(batch, prediction_key='predicted_image'):
    """Remove padded prediction values from reconstruction and GAN loss inputs.

    The model still sees padding as unknown. For the loss paths, use only the
    original map's mask and replace generated exterior pixels with the common
    GT background. Gradients cannot reach generated exterior pixels through
    these loss inputs. Discriminator receptive fields still see the canvas.
    """
    valid = batch['valid_region']
    supervised = dict(batch)
    supervised['mask'] = batch['mask'] * valid
    supervised['mask_for_losses'] = batch.get('mask_for_losses', batch['mask']) * valid
    supervised[prediction_key] = batch[prediction_key] * valid + batch['image'] * (1 - valid)
    return supervised
