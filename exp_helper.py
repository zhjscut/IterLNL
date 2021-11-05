import os
import time
import random
from multiprocessing import Pool

lr_list = [0.003]
ep_list = [5, 10, 20]
seed_list = [0, 1]
coeff_lossGD_list = [0.3]
percent_list = [1 - x for x in [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19]]
commands = []
# for lr in lr_list:
#     for ep in ep_list:
#         for seed in seed_list:
#             for coeff_lossGD in coeff_lossGD_list:
#                 # commands.append(f"CUDA_VISIBLE_DEVICES=0 python main.py configs/train/digit.yaml --source mnist --target usps --lr {lr} --max_epochs {ep} --seed {seed} --coeff_lossGD {coeff_lossGD}")
#                 commands.append(f"CUDA_VISIBLE_DEVICES=0 python main.py configs/train/digit.yaml --source usps --target mnist --lr {lr} --max_epochs {ep} --seed {seed} --coeff_lossGD {coeff_lossGD}")
#                 commands.append(f"CUDA_VISIBLE_DEVICES=0 python main.py configs/train/digit.yaml --source svhn --target mnist --lr {lr} --max_epochs {ep} --seed {seed} --coeff_lossGD {coeff_lossGD}")
for percent in percent_list:
    commands.append(f"CUDA_VISIBLE_DEVICES=6 python main.py configs/train/digit.yaml --source mnist --target usps --lr 0.003 --max_epochs 5 --seed 0 --percent {percent}")
    # commands.append(f"CUDA_VISIBLE_DEVICES=5 python main.py configs/train/digit.yaml --source usps --target mnist --lr 0.002 --max_epochs 40 --seed 0 --percent {percent}")
    # commands.append(f"CUDA_VISIBLE_DEVICES=3 python main.py configs/train/digit.yaml --source svhn --target mnist --lr 0.0002 --max_epochs 10 --seed 0 --percent {percent}")
    

def run(command):
    # second = random.randint(0, 30)
    # print(f"waiting {second} seconds to avoid using the same log file name")
    # time.sleep(second)
    os.system(command)

if __name__=='__main__':
    pool = Pool(6)

    for command in commands:
        pool.apply_async(func=run, args=(command,))

    pool.close()
    pool.join()
