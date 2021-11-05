# Copyright (c) Gorilla-Lab. All rights reserved.
import torch
import numpy as np
from ipdb import set_trace

from gorilla.solver import load_checkpoint
from utils import (to_cuda,
                    SampleFilter, CategoricalSampleFilter)
from solver import solver_IterLNL


class solver_IterLNL_digit(solver_IterLNL):
    # NOTE: After a source model is trained, put its filepath into the dict below
    source_models = {
        # "Digit_U2M": "log/Digit_U2M_seed_0_lr_0.1_ep_10/best_model.pth.tar",
        # "Digit_S2M": "models/source_models/Digit_S2M_seed_0_lr_0.01_ep_10_gamma_0.001/best_model.pth.tar",
        # "Digit_M2U": "models/source_models/Digit_M2U_seed_0_lr_2.0_ep_5_gamma_0.02/best_model.pth.tar",
        "Digit_U2M": "",
        "Digit_S2M": "",
        "Digit_M2U": "",
    }

    def solve(self):
        self.train_first_source_model()
        for self.loop in range(self.start_loop, self.cfg.loops):
            if self.loop != self.start_loop:
                # after resumption, firstly check if the acc is as expected
                self.get_pseudo_label_of_target_data()
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

    def train_first_source_model(self):
        checkpoint = self.cfg.get("checkpoint", "")
        if checkpoint:
            self.resume(checkpoint)
            return
        elif self.cfg.source_model != "":
            self.logger.info(f"load pre-trained model from: {self.cfg.source_model}")
            load_checkpoint(self.model, self.cfg.source_model)
            return

        self.epoch = 0
        self.best_acc = 0
        max_iters = self.cfg.max_epochs * self.iters_per_epoch
        print('iters in each epoch is: %d' % (self.iters_per_epoch))
        self.model.train()
        self.log_buffer.clear()

        self.lr_scheduler.step()
        for self.iter in range(1, max_iters + 1):
            self.timer.reset()
            # prepare the data for the model forward and backward
            samples = self.get_samples("train_src")
            if isinstance(samples, dict):
                data, gt = samples["img"], samples["target"]
            else:  # just for debugging
                data, gt = samples

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

    def train_a_new_target_model(self):
        # reset model parameters
        self.reset_params(self.model.G)
        self.reset_params(self.model.F)

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

        # self.iter = 0
        # self.acc = self.evaluate()
        # self.logger.info("  Test acc after iters {}: {:3f}".format(self.iter, acc.item()))
        self.model.train()
        # reset lr_scheduler
        self.lr_scheduler.last_epoch = 0
        self.lr_scheduler.step()

        self.actual_max_iters = int(self.cfg.max_iters + self.warmup_iters)
        self.log_buffer.clear()

        for self.iter in range(1, self.actual_max_iters + 1):
            self.timer.reset()

            if self.iter <= self.warmup_iters:
                percent = 1.0
            else:
                percent = 1 - self.tao * min(( (self.iter - self.warmup_iters - 1) / self.T ) ** self.cfg.pow, 1)
            samples = self.get_samples("train_tgt")
            data, gt = samples["img"], samples["target"]
            # print(data.max(), data.min(), data.mean(), data.abs().mean(), data.size(), gt[:10])
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
            # print("after:", data.size(), gt.size(), "expect:", round(percent * batch_size, 3))
            # if self.cfg.categorial_sample:
                # values, _ = self.lb_guessor.buffer_tensor.median(dim=1)
                # values = self.lb_guessor.buffer_tensor[5, :]
                # self.meta["median"] = ", ".join(["{:.2f}".format(acc) for acc in values.numpy()])
                # print("class-wise median of buffers: {}".format(", ".join(["{:.2f}".format(acc) for acc in values.numpy()])))

            self.log_buffer.update({"data_time": self.timer.since_start()})

            logit = self.model(data)["cate"]
            # compute the category loss of feature_source
            loss_C = self.criterions["CELoss"](logit, gt)
            # global mean probability vector entropy maximization
            loss_GD = self.criterions["GDLoss"](logit)
            # NOTE: we need to maximize loss_GD, so its symbol is negative
            loss_total = loss_C - loss_GD * self.cfg.coeff_lossGD
            self.log_buffer.update({
                "train/loss/total": loss_total.item(),
                "train/loss/loss_C": loss_C.item(),
                "train/loss/loss_GD": loss_GD.item()})

            self.optimizer.zero_grad()
            loss_total.backward()
            self.optimizer.step()
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
