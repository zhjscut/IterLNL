CUDA_VISIBLE_DEVICES=1 python main.py configs/train/visda.yaml --source T --target V --lr 0.003 --max_epochs 2 --seed 0 --coeff_lossGD 0.0 --percent 0.1
CUDA_VISIBLE_DEVICES=2 python main.py configs/train/visda.yaml --source T --target V --lr 0.003 --max_epochs 2 --seed 0 --coeff_lossGD 0.0 --percent 0.2
CUDA_VISIBLE_DEVICES=3 python main.py configs/train/visda.yaml --source T --target V --lr 0.003 --max_epochs 2 --seed 0 --coeff_lossGD 0.0 --percent 0.3
CUDA_VISIBLE_DEVICES=4 python main.py configs/train/visda.yaml --source T --target V --lr 0.003 --max_epochs 2 --seed 0 --coeff_lossGD 0.0 --percent 0.5
CUDA_VISIBLE_DEVICES=5 python main.py configs/train/visda.yaml --source T --target V --lr 0.003 --max_epochs 2 --seed 0 --coeff_lossGD 0.0 --percent 0.7
CUDA_VISIBLE_DEVICES=6 python main.py configs/train/visda.yaml --source T --target V --lr 0.003 --max_epochs 2 --seed 0 --coeff_lossGD 0.0 --percent 0.8
CUDA_VISIBLE_DEVICES=7 python main.py configs/train/visda.yaml --source T --target V --lr 0.003 --max_epochs 2 --seed 0 --coeff_lossGD 0.0 --percent 0.9
