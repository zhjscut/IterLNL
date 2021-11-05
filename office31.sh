######################## find a suitable source model ########################
# 78.2
CUDA_VISIBLE_DEVICES=0 python main.py configs/pretrain/office31.yaml --source A --target W --lr 0.0005

# 81.5
CUDA_VISIBLE_DEVICES=0 python main.py configs/pretrain/office31.yaml --source A --target D --lr 0.003

# 65.5
CUDA_VISIBLE_DEVICES=0 python main.py configs/pretrain/office31.yaml --source W --target A --lr 0.002 --gamma 0.001

# 99.4
CUDA_VISIBLE_DEVICES=0 python main.py configs/pretrain/office31.yaml --source W --target D --lr 0.005

# 64.9
CUDA_VISIBLE_DEVICES=0 python main.py configs/pretrain/office31.yaml --source D --target A --lr 0.001 --gamma 0.001

# 97.1
CUDA_VISIBLE_DEVICES=0 python main.py configs/pretrain/office31.yaml --source D --target W --lr 0.0075 --gamma 0.001

######################## standard experiment ########################
# 92.6 ± 0.3 with seeds 01
CUDA_VISIBLE_DEVICES=0 python main.py configs/train/office31.yaml --source A --target W --lr 0.003 --max_epochs 30

# 92.8 ± 0.3 with seeds 01
CUDA_VISIBLE_DEVICES=0 python main.py configs/train/office31.yaml --source A --target D --lr 0.003 --max_epochs 30

# 73.9 ± 0.2 with seeds 01
CUDA_VISIBLE_DEVICES=0 python main.py configs/train/office31.yaml --source W --target A --lr 0.003 --max_epochs 30

# 99.4 ± 0.0 with seeds 01
CUDA_VISIBLE_DEVICES=0 python main.py configs/train/office31.yaml --source W --target D --lr 0.003 --max_epochs 30

# 73.4 ± 0.5 with seeds 024
CUDA_VISIBLE_DEVICES=0 python main.py configs/train/office31.yaml --source D --target A --lr 0.003 --max_epochs 30

# 97.9 ± 0.3 with seeds 01
CUDA_VISIBLE_DEVICES=0 python main.py configs/train/office31.yaml --source D --target W --lr 0.003 --max_epochs 30

######################## supplementary material ########################
# self-training, 84.0 ± 1.5 with seeds 01
CUDA_VISIBLE_DEVICES=0 python main_self_training.py configs/train/office31.yaml --source A --target W --lr 0.003 --max_epochs 100 --coeff_lossGD 0.0 --warmup_epochs 20 --suffix selftraining
