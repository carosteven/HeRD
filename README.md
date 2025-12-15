# HeRD: Hierarchical Reinforcement Learning - Diffusion Policy

Official implementation for our paper:

> **Push Smarter, Not Harder: Hierarchical RL-Diffusion Policy for Efficient Nonprehensile Manipulation**  
> *Anonymous Authors*  
<!-- > [arXiv preprint / Conference TBD] -->

---

## 📦 Overview
HeRD Policy is a hierarchical framework that combines high-level reinforcement learning for spatial goal selection with a low-level diffusion policy for efficient trajectory generation in nonprehensile manipulation tasks.

<p align="center">
    <img src="./media/HeRD Flowchart.png"><br/>
    <em>The HeRD Policy framework.</em>
</p>

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone --recurse-submodules https://github.com/carosteven/HeRD.git
```

If you **already cloned the repo without submodules**, run:
```bash
git submodule update --init --recursive
```

### 2. (Optional) Create a Python environment
We recommend using Python $\geq$ 3.11
```bash
cd HeRD
python -m venv venv
source venv/bin/activate
```

### 3. Install submodules and dependencies
All dependencies are contained in the submodules, so no ```requirements.txt``` is needed.

Simply run:
```bash
bash scripts/setup_submodules.sh
```
This will:
1. Initialize and update all submodules
2. Install each one with all dependencies

If you prefer to do it manually:
```bash
git submodule update --init --recursive
pip install -e submodules/BenchNPIN
pip install -e submodules/spfa
pip install -e submodules/diffusionPolicy
```

---

## 🚀 Usage

Trained models available from [Google Drive](https://drive.google.com/drive/folders/1zEC1cGMKbA0MK3FadHyeTwx8H_K61ELw?usp=sharing).
Download the models to `models/diffusion_models/` and `models/rl_models/` as necessary.


### Evaluating
In `scripts/run_experiments.py` ensure `show=True`, `train.train_mode=False`, `evaulate.eval_mode=True`.

For each trial you wish to run, ensure you specify the environment type, RL model name, and whether to use the diffusion model from the lists `obs_configs`, `model_names`, and `diffusion_configs` in the `evaluation` settings.

Once the script is configured, run using:
```bash
python scripts/run_experiments.py
```

### Training Components
#### 1. Train RL Policy
In `scripts/run_experiments.py` ensure `train.train_mode=True`, and `evaulate.eval_mode=False`. We also suggest setting `show=False` for performance.

In the `train` settings, specify the `job_name`, and training timesteps `total_timesteps`. To train a RL policy that can generalize to all of the environment, we recommend `train.random_env=True`, which randomly chooses between `large_columns` and `large_divider` environments while training. To train in a specific environment, ensure `random_env=False` and specify the environment using `env.obstacle_config`.

To train a RL policy with a pretrained diffusion policy, set `diffusion.use_diffusion_policy=True` and specify the model name of the diffusion policy `diffusion.model_name`.

Once configured, run:
```bash
python scripts/run_experiments.py
```

#### 2. Train Diffusion Policy
```bash
python train_diffusion_policy.py --task boxdelivery_lowdim --exp_name diffusion_train_001
```

