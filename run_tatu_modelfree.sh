#!~/bin/bash
set -x
set -e


CUDA_VISIBLE_DEVICES=7 python train_tatu_modelfree.py --task "hopper-medium-replay-v2" --rollout-length 5 --critic_num 2 --seed 19  --algo-name "tatu_td3_bc"  --reward-penalty-coef 1.5 --pessimism-coef 3.5 --beta 1.5 --real-ratio 0.7