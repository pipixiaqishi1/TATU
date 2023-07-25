# Overview

This is the implementation of the paper: Uncertainty-driven Trajectory Truncation for Data Augmentation in Offline Reinforcement Learning

# Dependencies

- Python 3.7.13
- MuJoCo 2.3.0
- Gym 0.24.1
- D4RL 1.1
- PyTorch 1.12.0+cu113
- TensorFlow 2.10.0

# Usage

```
# for tatu+model-based offline RL
bash run_tatu_modelbased.sh
# for tatu+model-free offline RL
bash run_tatu_modelfree.sh
```

For different mujoco tasks, some hyperparametes may be diffrent. Please see the original paper for detailed hyperparameters.

