######################## find a suitable source model ########################
# 75.6
CUDA_VISIBLE_DEVICES=0 python main.py configs/pretrain/digit.yaml --source usps --target mnist --lr 0.1 --max_epochs 10 --gamma 0.001 --seed 0

# 72.6
CUDA_VISIBLE_DEVICES=0 python main.py configs/pretrain/digit.yaml --source mnist --target usps --lr 2.0 --max_epochs 5 --gamma 0.02 --seed 0

# 72.9
CUDA_VISIBLE_DEVICES=0 python main.py configs/pretrain/digit.yaml --source svhn --target mnist --lr 0.01 --max_epochs 10 --gamma 0.001 --seed 0

######################## standard experiment ########################
# 97.6 ± 0.1 with seed 01
CUDA_VISIBLE_DEVICES=0 python main.py configs/train/digit.yaml --source usps --target mnist --lr 0.003 --max_epochs 5 --seed 0

# 97.7 ± 0.0 with seed 01
CUDA_VISIBLE_DEVICES=0 python main.py configs/train/digit.yaml --source mnist --target usps --lr 0.002 --max_epochs 40 --seed 0

# 97.7 ± 0.1 with seed 01
CUDA_VISIBLE_DEVICES=0 python main.py configs/train/digit.yaml --source svhn --target mnist --lr 0.0002 --max_epochs 10 --seed 0

######################## supplementary material ########################
# self-training, 93.6 ± 0.3 with seeds 01
CUDA_VISIBLE_DEVICES=2 python main_self_training.py configs/train/digit.yaml --source svhn --target mnist --lr 0.0002 --max_epochs 30 --coeff_lossGD 0.0 --warmup_epochs 5 --suffix selftraining

# manually set noise level
CUDA_VISIBLE_DEVICES=6 python main.py configs/train/digit.yaml --source mnist --target usps --lr 0.003 --max_epochs 5 --seed 0 --percent 0.1
CUDA_VISIBLE_DEVICES=6 python main.py configs/train/digit.yaml --source usps --target mnist --lr 0.002 --max_epochs 40 --seed 0 --percent 0.1
CUDA_VISIBLE_DEVICES=6 python main.py configs/train/digit.yaml --source svhn --target mnist --lr 0.0002 --max_epochs 10 --seed 0 --percent 0.1
CUDA_VISIBLE_DEVICES=7 python main.py configs/train/office31.yaml --source A --target W --lr 0.003 --max_epochs 30 --seed 0 --percent 0.1
