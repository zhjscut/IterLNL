#!/usr/bin/env python
"""
Gorilla training script.

This script reads a given config file and runs the training or evaluation.
It is an entry point that is able to train standard models in gorilla.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use gorilla as a library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import argparse
import os
import sys
import time
from ipdb import set_trace

import torch
from torch.nn.parallel import DistributedDataParallel

from gorilla.core import get_world_size, get_local_rank, launch, set_random_seed
from gorilla.config import Config, merge_cfg_and_args, collect_logger
from gorilla.evaluation import DatasetEvaluators

from models import Model
from datasets import build_dataloaders
from gorilla2d.evaluation import ClsEvaluator
from solver import SolverSelfTraining, SolverSelfTrainingDigit


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = Config.fromfile(args.config_file)
    cfg = merge_cfg_and_args(cfg, args)
    # special treatment for some keys
    if args.seed is not None:
        cfg.seed = args.seed
    if args.lr is not None:
        cfg.optimizer.lr = args.lr
    if args.gamma is not None:
        cfg.lr_scheduler.gamma = args.gamma
    if "arch" in cfg and "arch_t" not in cfg:
        cfg.arch_t = cfg.arch
    return cfg


def main(args):
    cfg = setup(args)
    prefix = "{}_{}2{}".format(cfg.dataset,
                               cfg.source.capitalize()[0],
                               cfg.target.capitalize()[0])
    cfg.log_dir, logger = collect_logger(prefix=prefix,
                                         seed=cfg.seed,
                                         lr=cfg.optimizer.lr,
                                         ep=cfg.max_epochs,
                                         wep=cfg.warmup_epochs,
                                         thr=cfg.thr,
                                         suffix=cfg.suffix)
    logger.info(cfg)

    # set random seed to keep the result reproducible
    if "seed" in cfg:
        set_random_seed(cfg.seed, deterministic=True)

    dataloaders = build_dataloaders(cfg)
    # NOTE: Model() use cfg.num_classes generated in build_dataloader(), so Model()
    # should put behind build_dataloader()
    model = Model(cfg)

    evaluator = ClsEvaluator(class_wise=True, num_classes=cfg.num_classes)

    if cfg.dataset == "Digit":
        solver = SolverSelfTrainingDigit(model, dataloaders, evaluator, cfg)
    else:
        solver = SolverSelfTraining(model, dataloaders, evaluator, cfg)
    if args.eval_only:
        solver.evaluate()
        return

    distributed = get_world_size() > 1
    if distributed:  # NOTE: not updated
        model = DistributedDataParallel(model,
                                        device_ids=[get_local_rank()],
                                        broadcast_buffers=False)

    solver.solve()
    return


def default_argument_parser(epilog=None):
    """
    Create a parser with some common arguments used by detectron2 users.

    Args:
        epilog (str): epilog passed to ArgumentParser describing the usage.

    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser(
        epilog=epilog or f"""
Examples:

Run on single machine:
    $ {sys.argv[0]} cfg.yaml --num-gpus 8

Change some config options:
    $ {sys.argv[0]} cfg.yaml MODEL.WEIGHTS /path/to/weight.pth SOLVER.BASE_LR 0.001

Run on multiple machines:
    (machine0)$ {sys.argv[0]} --machine-rank 0 --num-machines 2 --dist-url <URL> [--other-flags]
    (machine1)$ {sys.argv[0]} --machine-rank 1 --num-machines 2 --dist-url <URL> [--other-flags]
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("config_file", help="path to config file")
    parser.add_argument("--eval-only",
                        action="store_true",
                        help="perform evaluation only")
    parser.add_argument("--num-gpus",
                        type=int,
                        default=1,
                        help="number of gpus *per machine*")
    parser.add_argument("--num-machines",
                        type=int,
                        default=1,
                        help="total number of machines")
    parser.add_argument("--machine-rank",
                        type=int,
                        default=0,
                        help="the rank of this machine (unique per machine)")

    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    port = 2**15 + 2**14 + hash(
        os.getuid() if sys.platform != "win32" else 1) % 2**14
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:{}".format(port),
        help="initialization URL for pytorch distributed backend. See "
        "https://pytorch.org/docs/stable/distributed.html for details.",
    )
    # network
    parser.add_argument('--not_pretrain', action='store_true', default=False,
        help='whether to load pretrained model parameter (default: load)')

    # optimizer
    parser.add_argument("--lr", type=float, help="learning rate")
    parser.add_argument('--max_epochs', type=int, help='maximum epochs')
    parser.add_argument("--gamma", type=float, help="gamma of inv lr_scheduler")

    # dataset
    parser.add_argument("--dataset", type=str, help="name of dataset")
    parser.add_argument("--source", type=str, help="source domain")
    parser.add_argument("--target", type=str, help="target domain")

    # method-specific
    parser.add_argument("--coeff_lossGD", type=float, help="coefficient of global diversity loss")

    # reproducibility
    parser.add_argument("--seed", type=int, help="random seed")

    # debugging
    parser.add_argument('--visualize', action='store_true', default=False,
                        help='whether visualize some indicators (default: no)')
    parser.add_argument('--draw_figure', action='store_true', default=False,
                        help='save some data to draw figure of the paper')

    # temporary
    parser.add_argument('--len_buffer', type=int, help='length of categorical sampler buffer')
    parser.add_argument('--thr', type=float, help='threshold for sample selection')
    parser.add_argument("--suffix", type=str, help="suffix of log file")
    parser.add_argument("--warmup_epochs", type=int, help="humber of warmup epochs")

    return parser


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args, ),
    )
