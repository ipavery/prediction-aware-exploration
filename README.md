# Prediction-Aware Indoor Exploration

Isaac Avery version

This repository is built upon the [PIPE Planner](https://github.com/castacks/pipe-planner) codebase and is used for our research on prediction-aware indoor robot exploration.

This is a standalone research repository. Clone this repository directly to reproduce or extend our work.

## Installation

### Clone this repository

```bash
git clone git@github.com:HUMANBASE-U/prediction-aware-exploration.git
cd prediction-aware-exploration
```

If you have not configured GitHub SSH access, use HTTPS instead:

```bash
git clone https://github.com/HUMANBASE-U/prediction-aware-exploration.git
cd prediction-aware-exploration
```

### Set up Conda Environment

Create an environment named `pipe` using the environment file inherited from the PIPE Planner codebase:

```bash
conda env create -n pipe -f lama/conda_env.yml
conda activate pipe
```

#### Build from Source to install 'range_libc'

Run these commands from the repository root:

```bash
cd range_libc/pywrapper

# Install build dependencies (if needed)
conda install -y cython

# Build and install
python setup.py install

# Return to the repository root and verify the installation
cd ../..
python -c "import range_libc; print('range_libc installed successfully')"
```

### Download pretrained prediction models (KTH dataset)
You can download pretrained models from this <a href="https://drive.google.com/drive/u/0/folders/1u9WZ9ftwaMbP-RVySuNSVEdUDV_x4Dw6">link</a>. Place the zip file under `pretrained_models` directory and unzip the file. 

From the repository root, run:

```bash
mv ~/Downloads/weights.zip pretrained_models/
cd pretrained_models
unzip weights.zip
cd ..
```

The `pretrained_model` directory and its subdirectories should be organized as below: 

    prediction-aware-exploration
    ├── pretrained_models
        ├── weights
            ├── big_lama
                ├── models
                    ├── best.ckpt
            ├── lama_ensemble
                ├── train_1
                    ├── models
                        ├── best.ckpt
                ├── train_2
                    ├── models
                        ├── best.ckpt
                ├── train_3
                    ├── models
                        ├── best.ckpt    

## Experiments
### Customize your own experiment
In configs/base.yaml, you can manually select map, starting pose, and planning method. All map information and starting points available at <a href="https://magenta-brow-f14.notion.site/25-Starting-Points-per-Map-28d544fc91ed80c5bbdbdc1fb49a13de?pvs=143">here</a>.


#### log_iou 
If true, your algorithm runs until reaching the maximum time step budget (1500 for small maps, 3000 for medium maps, and 6000 for large maps), or reaching the 95% IoU. It saves the IoU score per 20 time steps and when reaching 90% and 95% IoU. If false, the algorithms for designated 'mission_time' time step.

### Run the script
Run the 'explore.py' script as below:

```bash
cd scripts
python3 explore.py
```

## Upstream Project

This project builds on **PIPE Planner: Pathwise Information Gain with Map Predictions for Indoor Robot Exploration**, published at IROS 2025.

- [Original repository](https://github.com/castacks/pipe-planner)
- [Paper](https://arxiv.org/abs/2503.07504)
- [Project page](https://pipe-planner.github.io)
- [Video](https://youtu.be/oZEqbCBRn-I)


## Citation

If you use the PIPE Planner components of this repository, please cite the original work:

```bib
@inproceedings{baek2025pipe,
  author={Baek, Seungjae and Moon, Brady and Kim, Seungchan and Cao, Muqing and Ho, Cherie and Scherer, Sebastian and Jeon, Jeong Hwan},
  booktitle={2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)}, 
  title={PIPE Planner: Pathwise Information Gain with Map Predictions for Indoor Robot Exploration}, 
  year={2025},
  pages={7684-7691},
  doi={10.1109/IROS60139.2025.11246190}}
```
