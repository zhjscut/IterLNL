# Copyright (c) Gorilla-Lab. All rights reserved.
import os.path as osp
from ipdb import set_trace

import torch
from torch.utils.data.distributed import DistributedSampler

from gorilla import DATASETS, build_dataloader
from gorilla.config import Config
from gorilla.core import get_world_size, get_local_rank
from gorilla2d.datasets import build_dataloaders as _build_dataloaders
from gorilla2d.datasets.dataloader import _select_image_process, _select_image_process_digit

import numpy as np
import os
from .digits import load_svhn, load_mnist, load_usps

def build_dataloaders(cfg):
    r"""Build dataloaders with validation set."""
    if cfg.dataset == "Digit":
        return generate_dataloader_uda_digit(cfg)
    dataloaders = _build_dataloaders(cfg)
    Dataset = DATASETS.get(cfg.dataset)
    if cfg.dataset == "Digit":
        _, transform_test = _select_image_process_digit(cfg.transform_type)
    else:
        _, transform_test = _select_image_process(cfg.transform_type)

    cfg.train_set = True # only used for Digit dataset
    train_dataset_target_no_drop = Dataset(root=osp.join(cfg.data_root, cfg.dataset),
                                   domain=cfg.target,
                                   transform=transform_test,
                                   cfg=cfg)
    if cfg.distributed:
        sampler = DistributedSampler(train_dataset_target_no_drop,
                                     num_replicas=get_world_size(),
                                     rank=get_local_rank())
    else:
        sampler = None
    test_dataset_cfg = Config(dict(batch_size=cfg.samples_per_gpu_test,
                                   shuffle=False,
                                   num_workers=cfg.workers_per_gpu,
                                   pin_memory=False,
                                   sampler=sampler))
    train_loader_target_no_drop = build_dataloader(train_dataset_target_no_drop, test_dataset_cfg)

    dataloaders.update({"train_tgt_no_drop": train_loader_target_no_drop})

    if "use_val" in cfg and cfg.use_val:
        # NOTE: here validation data is included in target training data
        val_dataset_target = Dataset(root=osp.join(cfg.data_root, cfg.dataset),
                                    domain=cfg.target,
                                    transform=transform_test,
                                    cfg=cfg)

        if cfg.dataset == "VisDA2017":
            num_labeled_per_class = 30
        else:
            raise NotImplementedError(f"validation set construction of {cfg.dataset}")
        seed = 0
        root = osp.join(cfg.data_root, cfg.dataset)
        folder = "image_list"
        filename = os.path.join(root, folder, f"val_seed{seed}_labeled{num_labeled_per_class}.txt")
        if not os.path.isfile(filename):
            labels = np.array(val_dataset_target.targets)
            labeled_idx = np.array([], dtype=np.int32)
            state = np.random.get_state()
            np.random.seed(seed)
            for i in range(val_dataset_target.num_classes):
                idx = np.where(labels == i)[0]
                idx = np.random.choice(idx, num_labeled_per_class, False)
                labeled_idx = np.append(labeled_idx, idx)
            assert len(labeled_idx) == num_labeled_per_class * val_dataset_target.num_classes
            np.random.set_state(state)

            with open(filename, "w") as fp:
                fp.write(" ".join([str(idx) for idx in labeled_idx.tolist()]))
        else:
            with open(filename, "r") as fp:
                labeled_idx = [int(ind) for ind in fp.read().split(" ")]

        val_dataset_target.paths = np.array(val_dataset_target.paths)[labeled_idx]
        val_dataset_target.targets = np.array(val_dataset_target.targets)[labeled_idx]

        val_loader_target = build_dataloader(val_dataset_target, test_dataset_cfg)
        dataloaders.update({"val_tgt": val_loader_target})

    return dataloaders


class Loaded_dataset():
    r"""Dataset for whose data is loaded before the dataset's __init__,
    which means it just accepts a tensor containing full dataset.
    """
    def __init__(self, data, gt):
        self.data = data
        self.gt = gt

    def __getitem__(self, index):
        return dict(img=self.data[index], target=self.gt[index], path="")

    def __len__(self):
        return self.data.size(0)

    def reload(self, gt):
        # used in pesudo label learning
        self.gt = gt


def generate_dataloader_uda_digit(args):
    args.num_classes = 10
    usps = True if args.source == 'usps' or args.target == 'usps' else False
    scale = True if args.source == 'svhn' or args.target == 'svhn' else False

    def get_data(data, datapath, scale=False, usps=False, all_use=False):
        dataroot = os.path.join(datapath, data)
        if data == 'svhn':
            train_image, train_label, \
            test_image, test_label = load_svhn(dataroot)
        if data == 'mnist':
            train_image, train_label, \
            test_image, test_label = load_mnist(dataroot, scale=scale, usps=usps, all_use=all_use)
        if data == 'usps':
            train_image, train_label, \
            test_image, test_label = load_usps(dataroot, all_use=all_use)

        return train_image, train_label, test_image, test_label

    datapath = os.path.join("data", args.dataset)
    train_source, s_label_train, _, _ = get_data(args.source, datapath, scale=scale,
                                                 usps=usps, all_use=args.all_use)
    train_target, t_label_train, test_target, t_label_test = get_data(args.target, datapath, scale=scale,
                                                                      usps=usps, all_use=args.all_use)

    def split_val_and_test(data, gt, n_val):
        r"""n_val (int): number of validation samples per category"""
        num_classes = gt.max() + 1
        val_idx = []
        for i in range(num_classes):
            # make a class-balanced validation set
            val_idx += list(np.where(gt == i)[0][:n_val])
        test_idx = list(set(range(gt.shape[0])) - set(val_idx))

        img_val = data[val_idx]
        label_val = gt[val_idx]
        img_test = data[test_idx]
        label_test = gt[test_idx]

        return img_val, label_val, img_test, label_test

    if args.use_val:
        val_target, t_label_val, test_target, t_label_test = split_val_and_test(test_target, t_label_test, 10)

    ################### dataset ###################
    train_source = torch.from_numpy(train_source).float()
    s_label_train = torch.from_numpy(s_label_train).long()
    source_dataset = Loaded_dataset(train_source, s_label_train)

    train_target = torch.from_numpy(train_target).float()
    t_label_train = torch.from_numpy(t_label_train).long()
    target_dataset_pseudo = Loaded_dataset(train_target, t_label_train)

    if args.use_val:
        val_target = torch.from_numpy(val_target).float()
        t_label_val = torch.from_numpy(t_label_val).long()
        target_dataset_val = Loaded_dataset(val_target, t_label_val)

    test_target = torch.from_numpy(test_target).float()
    t_label_test = torch.from_numpy(t_label_test).long()
    target_dataset_test = Loaded_dataset(test_target, t_label_test)

    ################### dataloader ###################
    source_loader = torch.utils.data.DataLoader(source_dataset, batch_size=args.samples_per_gpu,
                                    num_workers=args.workers_per_gpu, shuffle=True, drop_last=True, pin_memory=True)
    if args.use_val:
        target_loader_val = torch.utils.data.DataLoader(target_dataset_val, batch_size=args.samples_per_gpu,
                                        num_workers=args.workers_per_gpu, shuffle=False, drop_last=False, pin_memory=True)
    target_loader_test = torch.utils.data.DataLoader(target_dataset_test, batch_size=args.samples_per_gpu,
                                    num_workers=args.workers_per_gpu, shuffle=False, drop_last=False, pin_memory=True)
    target_loader_pseudo = torch.utils.data.DataLoader(target_dataset_pseudo, batch_size=args.samples_per_gpu,
                                    num_workers=args.workers_per_gpu, shuffle=True, drop_last=True, pin_memory=True)
    target_loader_pseudo_no_drop = torch.utils.data.DataLoader(target_dataset_pseudo, batch_size=args.samples_per_gpu,
                                    num_workers=args.workers_per_gpu, shuffle=False, drop_last=False, pin_memory=True)

    dataloaders = {}
    dataloaders['train_src'] = source_loader
    if args.use_val:
        dataloaders['val_tgt'] = target_loader_val
    dataloaders['train_tgt'] = target_loader_pseudo
    dataloaders['train_tgt_no_drop'] = target_loader_pseudo_no_drop  # only used for sample selection
    dataloaders['test_tgt'] = target_loader_test

    return dataloaders
