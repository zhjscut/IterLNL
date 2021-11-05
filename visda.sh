######################## find a suitable source model ########################
# ResNet34, 44.6
CUDA_VISIBLE_DEVICES=0 python main.py configs/pretrain/visda.yaml --source T --target V --lr 0.0005 --max_epochs 3 --seed 0

# ResNet50, 48.8
CUDA_VISIBLE_DEVICES=0 python main.py configs/pretrain/visda.yaml --source T --target V --lr 0.002 --max_epochs 3 --seed 0

# ResNet101, 51.5
CUDA_VISIBLE_DEVICES=0 python main.py configs/pretrain/visda.yaml --source T --target V --lr 0.0003 --max_epochs 3 --seed 0

######################## standard experiment ########################
# (out-of-dated) ResNet34, 71.1 in L12
CUDA_VISIBLE_DEVICES=0 python main.py configs/train/visda.yaml --source T --target V --lr 0.003 --max_epochs 3 --seed 0

# (out-of-dated) ResNet50, 72.8 in L19
CUDA_VISIBLE_DEVICES=0 python main.py configs/train/visda.yaml --source T --target V --lr 0.003 --max_epochs 10 --seed 0

# ResNet101, 83.1 in L29
CUDA_VISIBLE_DEVICES=0 python main.py configs/train/visda.yaml --source T --target V --lr 0.003 --max_epochs 2 --seed 0 --coeff_lossGD 0.0

######################## other experiments ########################
# (out-of-dated) use validation set (set `use_val` in config file as True), 82.9 in L30
CUDA_VISIBLE_DEVICES=0 python main.py configs/train/visda.yaml --source T --target V --lr 0.001 --max_epochs 10 --seed 0

######################## ablation study (ResNet101) ########################
# with global diversity loss, 84.8 in L27
CUDA_VISIBLE_DEVICES=0 python main.py configs/train/visda.yaml --source T --target V --lr 0.003 --max_epochs 2 --seed 0 --coeff_lossGD 0.3

# w/o categorial sampling, 79.0 in L28 (uncomment 'categorial_sample: False' and the next three lines in config file)
CUDA_VISIBLE_DEVICES=0 python main.py configs/train/visda.yaml --source T --target V --lr 0.003 --max_epochs 2 --seed 0 --coeff_lossGD 0.3

# w/o categorial sampling and global diversity loss0, 72.4 in L30 (uncomment 'categorial_sample: False' and the next three lines in config file)
CUDA_VISIBLE_DEVICES=0 python main.py configs/train/visda.yaml --source T --target V --lr 0.003 --max_epochs 2 --seed 0 --coeff_lossGD 0.0

######################## supplementary material ########################
# (out-of-dated) ResNet34 to ResNet101, 71.7 in L10
CUDA_VISIBLE_DEVICES=0 python main.py configs/train/visda.yaml --source T --target V --lr 0.001 --max_epochs 4 --seed 0

# (out-of-dated) ResNet101 to ResNet34, 77.2 in L10
CUDA_VISIBLE_DEVICES=0 python main.py configs/train/visda.yaml --source T --target V --lr 0.0005 --max_epochs 10 --seed 0

######################## rebuttal ########################
# manually set noise level
CUDA_VISIBLE_DEVICES=0 python main.py configs/train/visda.yaml --source T --target V --lr 0.003 --max_epochs 2 --seed 0 --coeff_lossGD 0.0 --percent 0.1

# knowledge distillation
CUDA_VISIBLE_DEVICES=0 python main_kd.py configs/train/visda.yaml --source T --target V --lr 0.003 --max_epochs 50 --seed 0 --suffix KD

# label smoothing
CUDA_VISIBLE_DEVICES=0 python main_ls.py configs/train/visda.yaml --source T --target V --lr 0.003 --max_epochs 50 --seed 0 --suffix LS
