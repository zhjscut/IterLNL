# Copyright (c) Gorilla-Lab. All rights reserved.
import torch
import numpy as np
from ipdb import set_trace

from utils import to_cuda
from solver import solver_IterLNL_digit


class SolverSelfTrainingDigit(solver_IterLNL_digit):
    def solve(self):
        self.train_first_source_model()
        self.get_pseudo_label_of_target_data()
        self.evaluate()
        self.train_a_new_target_model()

        with open("log/summary.log", "a") as fp:
            fp.write(f"{self.cfg.log_dir}_self_training: acc {self.acc:.2f} best {self.best_acc:.2f} in epoch {self.best_epoch+1}\n")

    def train_a_new_target_model(self):
        # reset model parameters
        self.reset_params(self.model.G)
        self.reset_params(self.model.F)

        self.best_acc = 0
        self.epoch = 0

        self.model.train()
        # reset lr_scheduler
        self.lr_scheduler.last_epoch = 0
        self.lr_scheduler.step()

        self.actual_max_iters = int(self.cfg.max_iters + self.warmup_iters)
        self.log_buffer.clear()
        for self.iter in range(1, self.actual_max_iters + 1):
            self.timer.reset()

            samples = self.get_samples("train_tgt")
            data, gt = samples["img"], samples["target"]

            data = to_cuda(data)
            gt = to_cuda(gt)

            # the model should be trained under source model prediction at first,
            # otherwise it is unreliable to do self-training
            if self.iter > self.warmup_iters:
                # do self-training after warmup, gt should be replaced by target model prediction
                # .eval(), otherwise BN layer will be updated
                self.model.eval()
                with torch.no_grad():
                    logit = self.model(data)["cate"]
                predict_score = torch.softmax(logit, dim=1)
                gt_score = predict_score[np.arange(gt.size(0)), gt]
                valid = gt_score > self.cfg.thr
                gt_score, gt_predict = predict_score.topk(1, dim=1)
                gt_predict = gt_predict.squeeze()
                self.meta["score_statistic"] = [gt_score.max(), gt_score.min(), gt_score.mean()]
                self.meta["n_select"] = valid.sum().item()
                self.model.train()

                data = data[valid]
                gt = gt_predict[valid]
            else:
                self.meta["score_statistic"] = [-1, -1, -1]
                self.meta["n_select"] = -1

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
                            "score: max {:.2f} min {:.2f} mean {:.2f} select {} samples in latest iteration").format(
                        self.loop+1, self.epoch+1, self.max_epochs+self.cfg.warmup_epochs, self.iter, self.actual_max_iters,
                        out_tmp["batch_time"], out_tmp["data_time"], self.acc,
                        out_tmp["train/loss/total"], out_tmp["train/loss/loss_C"],
                        out_tmp["train/loss/loss_GD"],
                        *self.meta["score_statistic"],
                        self.meta["n_select"])

            self.logger.info(log_string)
            type(self).__base__.__base__.__base__.write(self)

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
