#!/bin/bash -l
#SBATCH --job-name=lama-test
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --qos=test
#SBATCH --output=logs/lama-test-%j.out
#SBATCH --error=logs/lama-test-%j.err
#SBATCH --mail-user=ipa28@byu.edu
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE

# usage:
# cd /home/ipa28/prediction-aware-exploration/lama
# sbatch test_lama.sbatch

python bin/smoke_satellite_inputs.py --data-dir "/home/ipa28/nobackup/autodelete/lama_map_pairs/train_3" \
        --checkpoint ../pretrained_models/weights/big_lama/models/best.ckpt \
        --device cuda:0