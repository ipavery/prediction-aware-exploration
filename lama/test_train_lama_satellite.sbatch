#!/bin/bash -l
#SBATCH --job-name=lama-train-test-satellite
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --qos=test
#SBATCH --output=logs/lama-train-test-satellite-%j.out
#SBATCH --error=logs/lama-train-test-satellite-%j.err
#SBATCH --mail-user=ipa28@byu.edu
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE

# usage: 
# cd /home/ipa28/prediction-aware-exploration/lama
# sbatch test_train_lama_satellite.sbatch

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE=offline
export TORCH_HOME="$HOME/.cache/torch"
DATA=/home/ipa28/nobackup/autodelete/lama_map_pairs
OUT=/home/ipa28/nobackup/autodelete/lama_satellite_training
mkdir -p "$OUT/experiments" "$OUT/tb_logs"
python -u bin/train.py -cn big-lama-satellite \
    location=kth_train_3 \
    location.data_root_dir="$DATA" \
    location.out_root_dir="$OUT/experiments" \
    location.tb_dir="$OUT/tb_logs" \
    pretrained_generator_path=/home/ipa28/prediction-aware-exploration/pretrained_models/weights/big_lama/models/best.ckpt \
    trainer.kwargs.gpus=1 data.batch_size=2 \
    trainer.kwargs.max_epochs=30 \
    trainer.kwargs.num_sanity_val_steps=0 \
    trainer.kwargs.val_check_interval=1.0 \
    data.num_workers=2 \
    +evaluator.fid=false \
    +evaluator.lpips=false \
    evaluator.integral_kind=null \
    trainer.checkpoint_kwargs.monitor=val_ssim_total_mean

# example for a quick test (1 epoch):
# python -u bin/train.py -cn big-lama-satellite \
#     location=kth_train_3 \
#     location.data_root_dir="$DATA" \
#     location.out_root_dir="$OUT/experiments" \
#     location.tb_dir="$OUT/tb_logs" \
#     pretrained_generator_path=/home/ipa28/prediction-aware-exploration/pretrained_models/weights/big_lama/models/best.ckpt \
#     trainer.kwargs.gpus=1 data.batch_size=2 \
#     trainer.kwargs.max_epochs=1 \
#     trainer.kwargs.limit_train_batches=1 \
#     +trainer.kwargs.limit_val_batches=1 \
#     trainer.kwargs.num_sanity_val_steps=0 \
#     trainer.kwargs.val_check_interval=1.0 \
#     data.num_workers=0 \
#     +evaluator.fid=false \
#     +evaluator.lpips=false \
#     evaluator.integral_kind=null \
#     trainer.checkpoint_kwargs.monitor=val_ssim_total_mean