#!/usr/bin/env python3
"""Small CUDA convolution/FFT forward-backward check before starting W&B."""
import sys
import importlib


def main():
    import torch

    print('Python executable: {}'.format(sys.executable), flush=True)
    import pytorch_lightning as pl
    import numpy as np
    if not pl.__version__.startswith('1.9.'):
        raise RuntimeError('Expected Lightning 1.9.x, found {}. Activate lama-h200.'.format(pl.__version__))
    # Import the real trainer before initializing W&B or allocating the full model.
    importlib.import_module('saicinpainting.training.trainers.default')
    print('LaMa trainer imports passed; Lightning={}, NumPy={}'.format(pl.__version__, np.__version__), flush=True)

    print('GPU preflight: PyTorch={}, built CUDA={}, cuDNN={}'.format(
        torch.__version__, torch.version.cuda, torch.backends.cudnn.version()), flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError('No usable CUDA device. This training job requires one GPU.')
    device = torch.cuda.current_device()
    print('Assigned GPU: {}; compute capability={}'.format(
        torch.cuda.get_device_name(device), torch.cuda.get_device_capability(device)), flush=True)
    if hasattr(torch.cuda, 'get_arch_list'):
        print('PyTorch compiled architectures: {}'.format(torch.cuda.get_arch_list()), flush=True)
    # Exercise both kinds of kernels required by LaMa. Do not reject an architecture
    # solely from get_arch_list: some builds can legitimately use forward-compatible PTX.
    image = torch.randn(1, 3, 32, 32, device='cuda', requires_grad=True)
    convolution = torch.nn.Conv2d(3, 4, kernel_size=3, padding=1).cuda()
    output = convolution(image)
    spectrum = torch.fft.rfft2(output, norm='ortho')
    restored = torch.fft.irfft2(spectrum, s=output.shape[-2:], norm='ortho')
    restored.square().mean().backward()
    torch.cuda.synchronize()
    print('GPU preflight passed: convolution and FFT forward/backward.', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print('GPU preflight FAILED: {}'.format(error), file=sys.stderr, flush=True)
        print('Confirm the job activated lama-h200 and uses the CUDA 11.8 PyTorch build. '
              'W&B and training were not started by this job.', file=sys.stderr, flush=True)
        raise SystemExit(2)
