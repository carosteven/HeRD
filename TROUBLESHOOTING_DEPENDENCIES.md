# Quick Fix: Missing Dependencies

## The Issue

When you run:
```bash
python3 eval_with_sampling_planner.py --num_eps 10
```

You see:
```
Error: Missing dependencies!
Details: No module named 'einops'
```

## The Solution

The HeRD project requires dependencies from the `diffusionPolicy` submodule to be installed.

### Option 1: Install All Dependencies (Recommended)

```bash
cd /Users/stevencaro/repos/HeRD
pip install -r requirements.txt
```

This installs:
- BenchNPIN
- spfa
- diffusionPolicy (with all its dependencies like einops, diffusers, etc.)

### Option 2: Install Just diffusionPolicy

```bash
cd /Users/stevencaro/repos/HeRD
pip install -e ./submodules/diffusionPolicy
```

### Option 3: Quick Test (If Dependencies Already Installed Elsewhere)

If you've already run the original HeRD code successfully, you likely have the dependencies installed in a virtual environment or conda environment. Make sure that environment is activated:

```bash
# If using conda
conda activate your-herd-env

# If using virtualenv
source /path/to/venv/bin/activate

# Then run
python3 eval_with_sampling_planner.py --num_eps 10
```

## Verification

After installing dependencies, verify with:

```bash
python3 -c "from einops import rearrange; print('✓ einops installed')"
python3 -c "from diffusionPolicy import DiffusionUnetLowdimPolicy; print('✓ diffusionPolicy installed')"
```

Both should print success messages.

## Why This Happens

The `SamplingPushingPlanner` itself only needs:
- numpy
- scipy  
- torch

But it integrates with `HeRDPolicy`, which imports `DiffusionUnetLowdimPolicy` from the diffusionPolicy submodule. Even though we replace the diffusion policy with our sampling planner, Python still needs to resolve the import at module load time.

## Once Dependencies Are Installed

Then you can run:

```bash
# Quick test
python3 eval_with_sampling_planner.py --num_eps 1

# Full evaluation  
python3 eval_with_sampling_planner.py --num_eps 10

# Challenge configuration
python3 eval_with_sampling_planner.py --num_eps 50 --config large_divider --verbose
```

## Alternative: Standalone Usage

If you want to use the planner without installing diffusionPolicy dependencies, you can use it standalone (though this requires more manual integration):

```python
from sampling_pushing_planner import SamplingPushingPlanner
import numpy as np
import torch

# This only requires numpy, scipy, torch
planner = SamplingPushingPlanner(horizon=32)

obs_dict = {
    'grid': torch.tensor(obstacle_map),
    'robot_pos': np.array([x, y]),
    'goal_pos': np.array([gx, gy]),
    'box_positions': [...],
    'receptacle_pos': np.array([rx, ry])
}

waypoints = planner.predict_action(obs_dict)['action']
```

---

**Bottom line**: Run `pip install -r requirements.txt` to fix the issue.
