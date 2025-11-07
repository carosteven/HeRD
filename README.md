# HeRD: Hierarchical Reinforcement Learning - Diffusion Policy

Official implementation for our paper:

> **Push Smarter, Not Harder: Hierarchical RL-Diffusion Policy for Efficient Nonprehensile Manipulation**  
> *Anonymous Authors*  
<!-- > [arXiv preprint / Conference TBD] -->

---

## 📦 Overview



---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone --recurse-submodules https://github.com/your-username/HeRD.git
```

If you **already cloned the repo without submodules**, run:
```bash
git submodule update --init --recursive
```

### 2. (Optional) Create a Python environment
We recommend using Python $\geq$ 3.10
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