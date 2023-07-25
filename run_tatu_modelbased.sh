#!~/bin/bash
set -x
set -e


CUDA_VISIBLE_DEVICES=7 python train_tatu_modelbased.py --task "hopper-medium-replay-v2" --rollout-length 5 --critic_num 2 --seed 32  --algo-name "tatu_mopo"  --reward-penalty-coef 1.0 --pessimism-coef 2.0 --real-ratio 0.05
