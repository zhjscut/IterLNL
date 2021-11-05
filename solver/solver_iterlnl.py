# Copyright (c) Gorilla-Lab. All rights reserved.
import os
import numpy as np
import math
from matplotlib import pyplot as plt
from ipdb import set_trace
import random
import pandas as pd
import seaborn as sns

import torch
from torch.cuda.amp import autocast as autocast, GradScaler
import torch.utils.model_zoo as model_zoo
from torch.nn.parallel import DistributedDataParallel

from gorilla.core import get_world_size, get_local_rank
from gorilla.utils import Timer, check_rand_state, display
from gorilla.solver import BaseSolver
from gorilla.solver import save_checkpoint, load_checkpoint

from utils import (to_cuda, model_urls,
                    accuracy_for_each_class_original,
                    SampleFilter, CategoricalSampleFilter)
from losses import GlobalDiverseLoss, ZeroOneLoss

class solver_IterLNL(BaseSolver):
    # NOTE: After a source model is trained, put its filepath into the dict below
    source_models = {
        # "VisDA2017_T2V": "models/source_models/VisDA2017_T2V_resnet34_seed_0_lr_0.0005_ep_3_gamma_0.0003/best_model.pth.tar",  # ResNet34
        # "VisDA2017_T2V": "models/source_models/VisDA2017_T2V_resnet50_seed_0_lr_0.002_ep_3_gamma_0.0003/best_model.pth.tar",  # ResNet50
        # "VisDA2017_T2V": "log/VisDA2017_T2V_seed_0_lr_0.0003_ep_3/best_model.pth.tar",  # ResNet101
        "VisDA2017_T2V": "",

        # "Office31_A2W": "log/Office31_A2W_seed_0_lr_0.0005_ep_100/best_model.pth.tar",
        # "Office31_A2D": "models/source_models/Office31_A2D_seed_0_lr_0.003_ep_100_gamma_0.0003/best_model.pth.tar",
        # "Office31_W2A": "models/source_models/Office31_W2A_seed_0_lr_0.002_ep_100_gamma_0.001/best_model.pth.tar",
        # "Office31_W2D": "models/source_models/Office31_W2D_seed_0_lr_0.005_ep_100_gamma_0.0003/best_model.pth.tar",
        # "Office31_D2A": "models/source_models/Office31_D2A_seed_0_lr_0.001_ep_100_gamma_0.001/best_model.pth.tar",
        # "Office31_D2W": "models/source_models/Office31_D2W_seed_0_lr_0.0075_ep_100_gamma_0.001/best_model.pth.tar",
        "Office31_A2W": "",
        "Office31_A2D": "",
        "Office31_W2A": "",
        "Office31_W2D": "",
        "Office31_D2A": "",
        "Office31_D2W": "",
    }

    def __init__(self, model, dataloaders, evaluator, cfg):
        super().__init__(model, dataloaders, cfg)
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
        self.criterions = {"CELoss": torch.nn.CrossEntropyLoss(),
                           "GDLoss": GlobalDiverseLoss(),
                           "01Loss": ZeroOneLoss(),
                           }
        self.best_acc = 0  # best acc in a loop
        self.best_acc_loop = 0  # best acc during loops
        self.loop = -1
        self.best_loop = -1
        self.best_epoch = -1
        self.timer = Timer()

        self.pseudo_labels_prev = np.zeros(len(self.dataloaders["train_tgt_no_drop"].dataset))  # record pseudo_labels in previous loop

        self.cfg.max_iters = self.cfg.max_epochs * self.iters_per_epoch
        self.warmup_iters = self.cfg.warmup_epochs * self.iters_per_epoch
        self.actual_max_iters = int(self.cfg.max_iters + self.warmup_iters)

        self.start_loop = -1

    def solve(self):
        self.train_first_source_model()
        for self.loop in range(self.start_loop, self.cfg.loops):
            if self.loop != self.start_loop:
                # after resumption, firstly check if the acc is as expected
                # self.logger.info("before get pseudo label")
                self.get_pseudo_label_of_target_data()
                # self.logger.info("after get pseudo label")
                if self.cfg.arch != self.cfg.arch_t and self.loop == self.start_loop + 1:
                    self.change_model()
                self.train_a_new_target_model()

            self.evaluate()

            if self.best_acc_loop < self.acc:
                self.best_acc_loop = self.acc
                self.best_loop = self.loop
            self.write("loop")

            if self.cfg.source_model == "": # give a stop after training source model
                with open("log/summary.log", "a") as fp:
                    fp.write(f"{self.cfg.log_dir}: acc {self.acc:.2f} best {self.best_acc:.2f} in epoch {self.best_epoch}\n")
                raise RuntimeError("Source model is trained, please run the IterLNL experiment next.")

            self.save("best_model",
                      meta=dict(loop=self.loop,
                                best_acc=self.best_acc,
                                rng_state_dict=self.rng_state_dict))

        with open("log/summary.log", "a") as fp:
            fp.write(f"{self.cfg.log_dir}: acc {self.acc:.2f} best {self.best_acc_loop:.2f} in loop {self.best_loop+1}\n")

    def get_ready(self):
        super().get_ready()
        # initialize data iterators
        data_iterators = {}
        for key in self.dataloaders.keys():
            data_iterators[key] = iter(self.dataloaders[key])
        self.data_iterators = data_iterators

        prefix = "{}_{}2{}".format(self.cfg.dataset,
                                   self.cfg.source.capitalize()[0],
                                   self.cfg.target.capitalize()[0])
        self.cfg.source_model = self.source_models[prefix]

        if "iters_per_epoch" in self.cfg:
            self.iters_per_epoch = self.cfg.iters_per_epoch
        elif self.cfg.source_model == "":
            self.iters_per_epoch = len(self.dataloaders['train_src'])
        else:
            self.iters_per_epoch = len(self.dataloaders["train_tgt"])
        self.logger.info('iters in each epoch is: %d' % (self.iters_per_epoch))

        if "max_iters" in self.cfg:
            self.max_iters = self.cfg.max_iters
            self.max_epochs = math.ceil(self.cfg.max_iters / self.iters_per_epoch)
        else:
            self.max_epochs = self.cfg.max_epochs
            self.max_iters = self.cfg.max_epochs * self.iters_per_epoch

        if "test_interval_epoch" in self.cfg:
            self.cfg.test_interval = max(int(self.cfg.test_interval_epoch * self.iters_per_epoch), 1)

        self.logger.info("length of dataset: {}".format(
            ", ".join(["{}: {}".format(k, len(v.dataset)) for k, v in self.dataloaders.items()])))
        self.logger.info("length of dataloader: {}".format(
            ", ".join(["{}: {}".format(k, len(v)) for k, v in self.dataloaders.items()])))

    def get_samples(self, data_name):
        assert data_name in self.dataloaders

        data_loader = self.dataloaders[data_name]
        data_iterator = self.data_iterators[data_name]
        assert data_loader is not None and data_iterator is not None, \
            "Check your dataloader of %s." % data_name

        try:
            # self.logger.info("before next...")
            sample = next(data_iterator)
            # self.logger.info("after next...")
        except StopIteration:
            # self.logger.info("before iter...")
            if self.cfg.distributed:
                # NOTE: set epoch, otherwise the shuffle order will not be changed
                data_loader.sampler.set_epoch(data_loader.sampler.epoch + 1)
            data_iterator = iter(data_loader)
            # self.logger.info("after iter...")
            # self.logger.info("before next...")
            sample = next(data_iterator)
            # self.logger.info("after next...")
            self.data_iterators[data_name] = data_iterator
        return sample

    def train_first_source_model(self):
        checkpoint = self.cfg.get("checkpoint", "")
        if checkpoint:
            self.resume(checkpoint)
            return
        elif self.cfg.source_model != "":
            self.logger.info(f"load pre-trained model from: {self.cfg.source_model}")
            load_checkpoint(self._model, self.cfg.source_model)
            return

        self.epoch = 0
        self.best_acc = 0
        max_iters = self.cfg.max_epochs * self.iters_per_epoch
        print('iters in each epoch is: %d' % (self.iters_per_epoch))
        self.model.train()
        self.log_buffer.clear()

        for self.iter in range(1, max_iters + 1):
            self.timer.reset()
            # prepare the data for the model forward and backward
            samples = self.get_samples("train_src")
            if self.cfg.transform_type == "strong":
                data, gt = samples["img_strong"], samples["target"]
            else:
                data, gt = samples["img"], samples["target"]
            if torch.cuda.is_available():
                data = data.cuda()
                gt = gt.cuda()

            self.log_buffer.update({"data_time": self.timer.since_start()})

            # compute the output of source domain and target domain
            outputs = self.model(data)

            # compute the category loss of feature_source
            loss_C = self.criterions["CELoss"](outputs["cate"], gt)

            loss_total = loss_C
            self.log_buffer.update({
                "train/loss/total": loss_total.item(),
                "train/loss/loss_C": loss_C.item()})
            self.evaluator.process(outputs["cate"], gt)

            self.optimizer.zero_grad()
            loss_total.backward()
            self.optimizer.step()

            self.evaluator.process(outputs["cate"], gt)

            self.log_buffer.update({"batch_time": self.timer.since_start()})

            if self.iter % self.cfg.log_interval == 0:
                self.acc = self.evaluator.evaluate()["acc"]  # become a member for using self.write()
                self.evaluator.reset()
                self.log_buffer.update({"train/acc": self.acc})

                self.log_buffer.average(self.cfg.log_interval)
                self.write("pretrain")

            self.lr_scheduler.step()
            # for param_group in self.optimizer.param_groups:
            #     print(self.iter, "lr of {}".format(param_group["name"]), param_group["lr"])

            if self.iter % self.cfg.save_interval == 0:
                self.save("latest_model",
                          meta=dict(epoch=self.epoch,
                                    rng_state_dict=self.rng_state_dict))

            if self.iter % self.cfg.test_interval == 0:
                self.evaluate()
                self.model.train()
                if self.acc > self.best_acc:
                    self.best_acc = self.acc
                    self.save("best_source_model",
                              meta=dict(loop=self.loop,
                                        best_acc=self.best_acc,
                                        rng_state_dict=self.rng_state_dict))

            if self.iter % self.iters_per_epoch == 0:
                print("lr after epoch {}:".format(self.epoch + 1))
                for group in self.optimizer.param_groups:
                    print(group["lr"])

                self.epoch += 1
        self.iter = 0  # reset self.iter

    def get_pseudo_label_of_target_data(self):
        self.model.eval()
        # score generated by source model is only used to decide 'percent' when not use validation set, rather than used in SampleFilter
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
        soft_label = torch.softmax(logits, dim=1)

        pseudo_label_score, pseudo_labels = torch.max(soft_label, dim=1)
        pseudo_label_score = pseudo_label_score.cpu()
        pseudo_labels = pseudo_labels.cpu().numpy()

        if self.cfg.draw_figure and self.cfg.dataset == "VisDA2017":
            # reference: https://blog.csdn.net/weixin_43550531/article/details/106676119
            confusion_matrix = np.zeros((self.cfg.num_classes, self.cfg.num_classes))
            for gt_label, pseudo_label in zip(self.dataloaders['test_tgt'].dataset.targets, pseudo_labels):
                confusion_matrix[gt_label, pseudo_label] += 1

            def draw_confusion_matrix(trans_mat, pic_name):
                trans_prob_mat = (trans_mat.T/np.sum(trans_mat, 1)).T

                label = ["{}".format(i) for i in range(1, trans_mat.shape[0]+1)]
                df = pd.DataFrame(trans_prob_mat, index=label, columns=label)

                # Plot
                plt.figure(figsize=(7.5, 7.5))
                sns.set(font_scale=1.2)
                sns.heatmap(df, cbar=False, cmap='Blues', fmt=".2f",
                            linewidths=0, annot=True, xticklabels=False, yticklabels=False)

                plt.tight_layout()
                plt.savefig(pic_name, transparent=True, dpi=800)

            draw_confusion_matrix(confusion_matrix, "source_only.pdf")
            raise RuntimeError("Drawing finished.")

        ## update the noisy labels of target data.
        if self.cfg.dataset == "Digit":
            self.dataloaders["train_tgt"].dataset.reload(pseudo_labels)
        else:
            self.dataloaders["train_tgt"].dataset.paths = filepaths
            self.dataloaders["train_tgt"].dataset.targets = pseudo_labels
        # NOTE: it is necessary to iter(dataloaders) at once for immediately apply the changes of dataset
        data_iterator = iter(self.dataloaders["train_tgt"])
        self.data_iterators["train_tgt"] = data_iterator

        if self.cfg.use_val:
            # get noise level τ by 1 - val_acc, and T for sample selection
            self.evaluator.reset()
            for _, data in enumerate(self.dataloaders["val_tgt"]):
                data, gt = data["img"].cuda(), data["target"].cuda()
                with torch.no_grad():
                    logit = self.model(data)["cate"]
                self.evaluator.process(logit, gt)

            if self.cfg.dataset == "VisDA2017":
                instance_acc = False
                if instance_acc:
                    val_acc = self.evaluator.evaluate()["acc"]
                else:  # class-wise acc
                    total_vector, correct_vector = self.evaluator.evaluate()["class_wise_acc"]
                    class_wise_acc = correct_vector / total_vector * 100
                    val_acc = class_wise_acc.mean().cpu().item()
                    self.logger.info("Labeled val class-wise avg {}, acc: {}".format(round(val_acc, 2),
                                        ", ".join(["{:.2f}".format(acc) for acc in class_wise_acc.cpu().numpy()])))
            else:
                val_acc = self.evaluator.evaluate()["acc"]

        else:
            thresh = self.cfg.thr
            ratio_greater_than_thresh = \
                float((pseudo_label_score > thresh).sum()) / pseudo_label_score.size(0)

            percent = ratio_greater_than_thresh

            rescale_type = "quad"
            if rescale_type == "quad":  # quadratic function
                k = self.cfg.quad_k
                if ratio_greater_than_thresh > 0.5:
                    percent = math.pow(-2 * (ratio_greater_than_thresh-1), 1/k) / (-2) + 1
                else:
                    percent = math.pow(2 * ratio_greater_than_thresh, 1/k) / 2
            elif rescale_type == "3_seg":  # three segments
                min_percent, max_percent = self.cfg.minp, self.cfg.maxp
                if ratio_greater_than_thresh < min_percent:
                    # select min_percent of samples
                    percent = min_percent
                elif ratio_greater_than_thresh > max_percent:
                    # ratio increase rapidly, so it need an upper bound
                    percent = max_percent
                else:
                    # select sample with a score greater than 'thresh'
                    percent = ratio_greater_than_thresh
            elif rescale_type == "2_seg":  # two segments
                min_percent, max_percent = self.cfg.minp, self.cfg.maxp
                if ratio_greater_than_thresh < min_percent:
                    # select min_percent of samples
                    percent = min_percent
                else:
                    # use a schedule to prevent using all the samples
                    percent = (max_percent - min_percent) / (1.0 - min_percent) * (ratio_greater_than_thresh - min_percent) + min_percent
            elif rescale_type == "none":
                pass
            else:
                raise NotImplementedError(f"rescale_type: {rescale_type}")

        if "preset_percent" in self.cfg:
            min_percent = self.cfg.minp
            max_percent = self.cfg.maxp
            percent = self.loop / (self.cfg.loops - 1) * (max_percent - min_percent) + min_percent
            self.tao = 1 - percent
        else:
            if "percent" in self.cfg:
                # use manual τ without val_acc
                self.tao = 1 - self.cfg.percent
            elif self.cfg.use_val:
                # estimate τ with val_acc
                self.tao = (1 - val_acc / 100) * self.cfg.k_tao
            else:
                # estimate τ with the ratio of high score prediction
                self.tao = (1 - percent) * self.cfg.k_tao
                # self.tao = 1 - self.acc_ins / 100  # temporary code, only used for ablation study of using gt to estimate noisy label

        self.T = self.cfg.pos_end * self.cfg.max_iters
        print_string = "tao: {}, T: {}".format(self.tao, self.T)
        if self.cfg.use_val:
            print_string += " val_acc: {}".format(val_acc)
        else:
            print_string += " ratio greater than thresh: {}".format(ratio_greater_than_thresh)
        self.logger.info(print_string)

    def change_model(self):
        # NOTE: change_model() does not support distributed training
        # for different source and target model architecture
        print_string = f"change model from {self.cfg.arch} to {self.cfg.arch_t}"
        self.logger.info(print_string)
        from models import Model
        self.cfg.arch = self.cfg.arch_t
        self.model = Model(self.cfg)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        from gorilla import build_optimizer
        self.optimizer = build_optimizer(self.model, self.cfg.optimizer)

    def train_a_new_target_model(self):
        # reset model parameters
        if self.cfg.not_pretrain:
            self.reset_params(self._model.G)
        else:
            self.load_pretrained_dict(self._model.G)
        self.reset_params(self._model.F)

        # reset label guessor (because it has a cumulated parameter)
        if self.cfg.categorial_sample:
            ## the buffer is initialized here.
            self.lb_guessor = CategoricalSampleFilter(num_class=self.cfg.num_classes, len_buffer=self.cfg.len_buffer)
        else:
            # init_thresh = 0.0 is corresponding to the schedule of selecting all data at first
            self.lb_guessor = SampleFilter(alpha=self.cfg.beta, init_thresh=0.0,
                thresh_mode=self.cfg.thresh_mode, mode=self.cfg.sample_filter_mode)

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

            if self.iter <= self.warmup_iters:
                percent = 1.0
            else:
                percent = 1 - self.tao * min(( (self.iter - self.warmup_iters - 1) / self.T ) ** self.cfg.pow, 1)
            # self.logger.info("before get_samples...")
            samples = self.get_samples("train_tgt")
            # self.logger.info("after get_samples...")
            # idx is used in distillation loss
            data, gt, idx = samples["img"], samples["target"], samples["idx"]
            # display("data", data)

            batch_size = data.size(0)

            data = to_cuda(data)
            gt = to_cuda(gt)
            # print("before:", data.size(), gt.size())

            # .eval(), otherwise BN layer will be updated
            self.model.eval()
            with torch.no_grad():
                logit = self.model(data)["cate"]
            predict_score = torch.softmax(logit, dim=1)
            gt_score = predict_score[np.arange(gt.size(0)), gt]
            if self.cfg.categorial_sample:
                valid, score_statistic = self.lb_guessor(gt_score, gt, percent)
            else:
                valid, score_statistic = self.lb_guessor(gt_score, percent)
            self.meta["score_statistic"] = score_statistic
            self.meta["n_select"] = valid.sum().item()
            self.meta["n_expect"] = round(percent * batch_size, 3)
            self.model.train()
            if valid.sum(0) <= 1:
                self.logger.info("number of selected samples is less than 2, pass this batch")
                continue

            data = data[valid]
            gt = gt[valid]
            idx = idx[valid]
            # print("after:", data.size(), gt.size(), "expect:", round(percent * batch_size, 3))

            self.log_buffer.update({"data_time": self.timer.since_start()})

            # manually call __enter__() and __exit__() to replace 'with' statement and only write code block once
            if self.cfg.amp:
                _autocast = autocast()
                _autocast.__enter__()
            # self.logger.info("before forward...")
            logit = self.model(data)["cate"]
            # self.logger.info("after forward...")
            # compute the category loss of feature_source
            loss_C = self.criterions["CELoss"](logit, gt)
            # global mean probability vector entropy maximization
            loss_GD = self.criterions["GDLoss"](logit)
            # NOTE: we need to maximize loss_GD, so its symbol is negative
            loss_total = loss_C - loss_GD * self.cfg.coeff_lossGD
            self.optimizer.zero_grad()

            # self.logger.info("before backward...")
            if self.cfg.amp:
                _autocast.__exit__()
                self.scaler.scale(loss_total).backward()
                self.scaler.unscale_(self.optimizer)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss_total.backward()
                self.optimizer.step()
            # self.logger.info("before backward...")

            self.log_buffer.update({
                "train/loss/total": loss_total.item(),
                "train/loss/loss_C": loss_C.item(),
                "train/loss/loss_GD": loss_GD.item()})
            self.evaluator.process(logit, gt)

            self.log_buffer.update({"batch_time": self.timer.since_start()})
            self.lr_scheduler.step()

            if self.iter % self.cfg.log_interval == 0:
                self.acc = self.evaluator.evaluate()["acc"]  # become a member for using self.write()
                self.evaluator.reset()

                if self.cfg.categorial_sample:
                    values, _ = self.lb_guessor.buffer_tensor.median(dim=1)
                    self.meta["median"] = ", ".join(["{:.2f}".format(acc) for acc in values.numpy()])

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

    def train(self):
        # not used
        pass

    def evaluate(self):
        self.model.eval()
        self.log_buffer.clear()

        self.timer.reset()
        # reset evaluator to ensure correctness of evaluation, and it may lost some training acc data
        # because they share one evaluator
        self.evaluator.reset()

        with torch.no_grad():
            # fix the issue that self.evaluate() impacts the data order of training dataloader, which changes the experimental result
            rng_state = torch.get_rng_state()
            for _, data in enumerate(self.dataloaders["test_tgt"]):
                # self.logger.info("test a batch")
                if isinstance(data, dict):
                    target_data, target_gt = data["img"], data["target"]
                else:  # just for debugging
                    target_data, target_gt = data
                if torch.cuda.is_available():
                    target_data = target_data.cuda()
                    target_gt = target_gt.cuda()

                outputs_target = self.model(target_data)["cate"]

                self.evaluator.process(outputs_target, target_gt)
            torch.set_rng_state(rng_state)

        self.acc = self.evaluator.evaluate()["acc"]  # become a member for using self.write()
        total_vector, correct_vector = self.evaluator.evaluate()["class_wise_acc"]
        class_wise_acc = correct_vector / total_vector * 100
        self.logger.info("Eval class-wise avg {}, acc: {}".format(round(class_wise_acc.mean().cpu().item(), 2),
                            ", ".join(["{:.2f}".format(acc) for acc in class_wise_acc.cpu().numpy()])))
        if self.cfg.dataset == "VisDA2017":
            self.acc = class_wise_acc.mean().cpu().item()
            # self.acc_ins = self.evaluator.evaluate()["acc"]  # temporary code, only used for ablation study of using gt to estimate noisy label

        self.evaluator.reset()
        self.log_buffer.update({"eval/acc": self.acc})
        if self.best_acc < self.acc:
            self.best_acc = self.acc
            self.best_epoch = self.epoch

            self.save("best_model",
                      meta=dict(epoch=self.epoch,
                                best_acc=self.best_acc))

        self.write("eval")

    def write(self, mode):
        r"""
        Write infos into log file and tensorboard.
        mode can be 'train' or 'eval'.
        """
        if mode == "pretrain":
            out_tmp = self.log_buffer.output
            log_string = ("Pre-Tr ep [{}/{}]  it [{}/{}]  BT {:.3f}  DT {:.3f}   acc {:.3f}\n"
                            "loss_total {:.3f}").format(
                        self.epoch+1, self.max_epochs, self.iter, self.max_iters,
                        out_tmp["batch_time"], out_tmp["data_time"],
                        self.acc, out_tmp["train/loss/total"])
            self.logger.info(log_string)

        elif mode == "train":
            out_tmp = self.log_buffer.output
            log_string = ("Tr loop {}  ep [{}/{}]  it [{}/{}]  BT {:.3f}  DT {:.3f}   acc {:.3f}\n"
                            "loss_total {:.3f}   loss_C {:.3f}   loss_GD {:.3f}\n"
                            "score: max {:.2f} min {:.2f} mean {:.2f} select {} samples in latest iteration, "
                            "expected {} samples").format(
                        self.loop+1, self.epoch+1, self.max_epochs+self.cfg.warmup_epochs, self.iter, self.actual_max_iters,
                        out_tmp["batch_time"], out_tmp["data_time"], self.acc,
                        out_tmp["train/loss/total"], out_tmp["train/loss/loss_C"],
                        out_tmp["train/loss/loss_GD"],
                        *self.meta["score_statistic"],
                        self.meta["n_select"], self.meta["n_expect"])
            if self.cfg.categorial_sample:
               log_string += ("\nclass-wise median of buffers: {}".format(self.meta["median"]))

            self.logger.info(log_string)

            super().write()

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

        elif mode == "loop":
            print_string = "  Test acc after loop {}: {:.2f}, the best is {:.2f} in loop {}".format(
                self.loop + 1, self.acc,
                self.best_acc_loop, self.best_loop + 1)
            self.logger.info(print_string + "\n")
        else:
            raise NotImplementedError("mode: {} for Solver.write()".format(mode))

    def save(self, filename, meta=None):
        filename = "{}.pth.tar".format(filename)
        dir_save_file = os.path.join(self.cfg.log_dir, filename)
        save_checkpoint(self.model,
                        dir_save_file,
                        self.optimizer,
                        self.lr_scheduler,
                        meta=meta)

    def resume(self, checkpoint):
        super().resume(checkpoint)
        self.epoch = self.meta["epoch"] + 1
        self.iter = self.meta["epoch"] * self.iters_per_epoch
        # NOTE: this version still can not reproduce the result of running without resume
        self.start_loop = self.meta['loop']
        rng_states = self.meta["rng_state_dict"]
        random.setstate(rng_states["random"])
        np.random.set_state(rng_states["numpy"])
        torch.set_rng_state(rng_states["torch"])
        torch.cuda.set_rng_state_all(rng_states["cuda"])

    @property
    def rng_state_dict(self):
        return {"random": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                # get_rng_state_all() only contain device in CUDA_VISIBLE_DEVICES rather than all on server, which is safe
                "cuda": torch.cuda.get_rng_state_all()
                }

    def reset_params(self, module):
        for m in module.modules():
            if id(m) != id(module):
                # in case of some modules without reset_parameters() such as ReLU
                if hasattr(m, "reset_parameters"):
                    m.reset_parameters()

    def load_pretrained_dict(self, module):
        pretrained_dict = model_zoo.load_url(model_urls[self.cfg.arch])
        model_dict = module.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items()
                        if k in model_dict}
        model_dict.update(pretrained_dict)
        module.load_state_dict(model_dict)

    def get_class_wise_acc_and_log(self, soft_label, gt, desc):
        r"""desc (str): Description in front of logging information."""
        # class-wise predict number and class-wise correct number
        predict_nums = torch.zeros(self.cfg.num_classes)
        correct_nums = torch.zeros(self.cfg.num_classes)
        predict_nums, correct_nums = accuracy_for_each_class_original(soft_label, gt, predict_nums, correct_nums)
        class_wise_acc = correct_nums / predict_nums
        with open(os.path.join(self.cfg.log_dir, 'log.txt'), 'a') as log:
            log.write("\n{} class-wise avg {}, acc: {}".format(desc, round(class_wise_acc.mean().cpu().item(), 2),
                        ", ".join(["{:.2f}".format(acc) for acc in class_wise_acc.cpu().numpy()])))
        return class_wise_acc.mean().cpu().item()
