# Copyright (c) Gorilla-Lab. All rights reserved.
from ipdb import set_trace
import numpy as np

import torch
from torch.cuda.amp import autocast as autocast, GradScaler

from gorilla.utils import Timer

from solver import solver_IterLNL
from utils import to_cuda
from losses import DistillationLoss

class SolverKD(solver_IterLNL):
    # NOTE: After a source model is trained, put its filepath into the dict below
    source_models = {
        "VisDA2017_T2V": "models/source_models/VisDA2017_T2V_resnet101_seed_0_lr_0.0003_ep_3_gamma_0.0003/best_model.pth.tar",  # ResNet101
    }

    def __init__(self, model, dataloaders, evaluator, cfg):
        # NOTE: use BaseSolver's init function, not use solver_IterLNL one
        super(solver_IterLNL, self).__init__(model, dataloaders, cfg)
        if self.cfg.distributed:
            # self.model will be used in model forward and backward, and
            # self._model will be used as `self.model` in other situation of not paralleled version
            self._model = self.model.module
        else:
            self._model = self.model
        # NOTE: currently it is not ensured reliable to run amp and DDP simultaneously
        if self.cfg.amp:
            self.scaler = GradScaler()

        self.evaluator = evaluator
        self.criterions = {"DistillLoss": DistillationLoss(),
                           }
        self.best_acc = 0  # best acc in a loop
        self.loop = -1
        self.best_epoch = -1
        self.timer = Timer()

        self.pseudo_labels_prev = np.zeros(len(self.dataloaders["train_tgt_no_drop"].dataset))  # record pseudo_labels in previous loop

        # KD do not need warmup, here just keep the change from iterlnl code as less as possible
        self.cfg.max_iters = self.cfg.max_epochs * self.iters_per_epoch
        self.warmup_iters = self.cfg.warmup_epochs * self.iters_per_epoch
        self.actual_max_iters = int(self.cfg.max_iters + self.warmup_iters)

        self.start_loop = -1

    def solve(self):
        self.train_first_source_model()
        # self.evaluate()
        self.get_soft_label_of_target_data()
        self.train_a_new_target_model()

        with open("log/summary.log", "a") as fp:
            fp.write(f"{self.cfg.log_dir}: acc {self.acc:.2f} best {self.best_acc:.2f} in epoch {self.best_epoch}\n")

    def get_soft_label_of_target_data(self):
        self.model.eval()
        logits = torch.zeros(len(self.dataloaders["train_tgt_no_drop"].dataset), self.cfg.num_classes).cuda()  # .cuda(): 20M gpu memory for 3s faster
        batch_size = self.dataloaders["train_tgt_no_drop"].batch_size
        filepaths = []
        for i, data in enumerate(self.dataloaders["train_tgt_no_drop"]):
            target_data, filepath = data["img"], data["path"]
            if torch.cuda.is_available():
                target_data = target_data.cuda()
            with torch.no_grad():
                logit = self.model(target_data)["cate"]
            logits[i * batch_size: i * batch_size + logit.size(0), :] = logit
            filepaths += filepath
        self.soft_label = torch.softmax(logits, dim=1)

    def train_a_new_target_model(self):
        # reset model parameters
        if self.cfg.not_pretrain:
            self.reset_params(self._model.G)
        else:
            self.load_pretrained_dict(self._model.G)
        self.reset_params(self._model.F)

        self.best_acc = 0
        self.epoch = 0

        self.model.train()
        # reset lr_scheduler
        self.lr_scheduler.last_epoch = 0
        self.lr_scheduler.step()

        self.actual_max_iters = int(self.cfg.max_iters + self.warmup_iters)
        self.log_buffer.clear()
        for self.iter in range(1, self.actual_max_iters + 1):
            # self.logger.info(f"Iter {self.iter}")
            # check_rand_state(self.logger)
            self.timer.reset()

            # self.logger.info("before get_samples...")
            samples = self.get_samples("train_tgt")
            # self.logger.info("after get_samples...")
            # idx is used in distillation loss
            data, gt, idx = samples["img"], samples["target"], samples["idx"]
            # display("data", data)

            batch_size = data.size(0)

            data = to_cuda(data)
            gt = to_cuda(gt)

            self.log_buffer.update({"data_time": self.timer.since_start()})

            # manually call __enter__() and __exit__() to replace 'with' statement and only write code block once
            if self.cfg.amp:
                _autocast = autocast()
                _autocast.__enter__()
            
            logit = self.model(data)["cate"]
            # compute the category loss of feature_source
            loss_Distill = self.criterions["DistillLoss"](self.soft_label[idx, :], logit)
            # NOTE: we need to maximize loss_GD, so its symbol is negative
            loss_total = loss_Distill
            self.optimizer.zero_grad()

            if self.cfg.amp:
                _autocast.__exit__()
                self.scaler.scale(loss_total).backward()
                self.scaler.unscale_(self.optimizer)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss_total.backward()
                self.optimizer.step()

            self.log_buffer.update({
                "train/loss/total": loss_total.item()})
            self.evaluator.process(logit, gt)

            self.log_buffer.update({"batch_time": self.timer.since_start()})
            self.lr_scheduler.step()

            if self.iter % self.cfg.log_interval == 0:
                self.acc = self.evaluator.evaluate()["acc"]  # become a member for using self.write()
                self.evaluator.reset()

                self.log_buffer.average(self.cfg.log_interval)
                self.write("train")

            if self.iter % self.cfg.save_interval == 0:
                self.save("latest_model", meta=dict(loop=self.loop,
                                                    iters=self.iter))

            if self.iter % self.iters_per_epoch == 0:
                self.logger.info("lr after epoch {}:".format(self.epoch+1))
                for group in self.optimizer.param_groups:
                    self.logger.info(group["lr"])

                self.epoch += 1

            if self.iter % self.cfg.test_interval == 0:
                self.evaluate()
                self.model.train()

    def write(self, mode):
        r"""
        Write infos into log file and tensorboard.
        mode can be 'train' or 'eval'.
        """
        if mode == "train":
            out_tmp = self.log_buffer.output
            log_string = ("Tr loop {}  ep [{}/{}]  it [{}/{}]  BT {:.3f}  DT {:.3f}   acc {:.3f}\n"
                            "loss_total {:.3f}\n").format(
                        self.loop+1, self.epoch+1, self.max_epochs+self.cfg.warmup_epochs, self.iter, self.actual_max_iters,
                        out_tmp["batch_time"], out_tmp["data_time"], self.acc,
                        out_tmp["train/loss/total"])

            self.logger.info(log_string)

            super(solver_IterLNL, self).write()

        elif mode == "eval":
            print_string = ("Te it [{}/{}]  Time {:.3f}   "
                            "Target acc {:.3f}  Best acc so far {:.3f} in epoch {}").format(
                                self.iter, self.actual_max_iters,
                                self.timer.since_last(),
                                self.acc, self.best_acc, self.best_epoch)
            self.logger.info(print_string + "\n")
            self.tb_writer.add_scalars(
                "acc", {
                    "test": self.acc,
                    "best_acc": self.best_acc
                }, self.iter)

        else:
            raise NotImplementedError("mode: {} for Solver.write()".format(mode))

