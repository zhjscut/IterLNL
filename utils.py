
import os
import copy
import math
import shutil
from typing import Dict, List, Tuple

import numpy as np
import imageio
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

import torch
import torch.nn as nn
import torch.nn.functional as F
# from spherecluster import SphericalKMeans
import ipdb


def accuracy(output, target):
    r"""Computes the precision"""
    batch_size = target.size(0)
    _, pred = output.topk(1, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    correct = correct[:1].view(-1).float().sum(0, keepdim=True)
    res = correct.mul_(100.0 / batch_size)
    return res


def get_prediction_with_uniform_prior(soft_prediction):
    # if classifier give a bias on a certain class, most of samples will be predicted to this class.
    # The uniform prior operation can reduce the unbalance of this problem, and keep the prediction near balanced.
    # NOTE: soft_prediction contain the whole dataset, not a batch
    soft_prediction_uniform = soft_prediction / soft_prediction.sum(0, keepdim=True).pow(0.5)
    soft_prediction_uniform /= soft_prediction_uniform.sum(1, keepdim=True)
    return soft_prediction_uniform


def get_labels_from_classifier_prediction(target_u_prediction_matrix, T, gt_label):
    # NOTE: target_u_prediction_matrix contain the whole dataset, not a batch
    target_u_prediction_matrix_withT = target_u_prediction_matrix / T
    soft_label_fc = torch.softmax(target_u_prediction_matrix_withT, dim=1)
    scores, hard_label_fc = torch.max(soft_label_fc, dim=1)

    soft_label_uniform_fc = get_prediction_with_uniform_prior(soft_label_fc)
    scores_uniform, hard_label_uniform_fc = torch.max(soft_label_fc, dim=1)

    acc_fc = accuracy(soft_label_fc, gt_label)
    acc_uniform_fc = accuracy(soft_label_uniform_fc, gt_label)
    print('acc of fc is: %3f' % (acc_fc))
    print('acc of fc with uniform prior is: %3f' % (acc_uniform_fc))

    return soft_label_fc, soft_label_uniform_fc, hard_label_fc, hard_label_uniform_fc, acc_fc, acc_uniform_fc


class LabelGuessor(object):

    def __init__(self, thresh):
        self.thresh = thresh

    def __call__(self, G, F, ims):
        org_state_G = {
            k: v.clone().detach()
            for k, v in G.state_dict().items()
        }
        org_state_F = {
            k: v.clone().detach()
            for k, v in F.state_dict().items()
        }
        is_train = G.training
        with torch.no_grad():
            G.train()
            F.train()
            all_probs = []
            logits = F(G(ims))
            probs = torch.softmax(logits, dim=1)
            scores, lbs = torch.max(probs, dim=1)
            idx = scores > self.thresh
            lbs = lbs[idx]

        G.load_state_dict(org_state_G)
        F.load_state_dict(org_state_F)
        if is_train:
            G.train()
            F.train()
        else:
            G.eval()
            F.eval()
        return lbs.detach(), idx


class LabelGuessor_dymaticT(object):

    def __init__(self, alpha=0.9, init_thresh=1.0, mode='train'):
        self.alpha = alpha
        self.mode = mode
        self.thresh = init_thresh

    def __call__(self, G, F, ims, percent):
        org_state_G = {
            k: v.clone().detach()
            for k, v in G.state_dict().items()
        }
        org_state_F = {
            k: v.clone().detach()
            for k, v in F.state_dict().items()
        }
        is_train = G.training
        with torch.no_grad():
            if self.mode == 'train':
                G.train()
                F.train()
            elif self.mode == 'eval':
                G.eval()
                F.eval()

            logits = F(G(ims))
            probs = torch.softmax(logits, dim=1)
        scores, lbs = torch.max(probs, dim=1)
        sorted_values, _ = torch.sort(scores, descending=True)  ### from large to small

        num_batch = logits.size(0)
        selected = num_batch * percent
        if selected > num_batch:
            selected = num_batch
        round_number = round(selected)
        ceil_number = math.ceil(selected)
        ## get the threshold in this batch level
        if round_number == ceil_number:
            if round_number == 0:
                thresh_batch = 1.0
            else:
                thresh_batch = sorted_values[round_number - 1]
        else:
            ## round value >= ceil value
            if round_number == 0:
                round_value = 1.0
            else:
                round_value = sorted_values[round_number - 1]
            ceil_value = sorted_values[ceil_number - 1]
            ## get the thresh_value with interpolation
            thresh_batch = round_value - (selected - round_number) * (round_value - ceil_value)

        self.thresh = self.thresh * self.alpha + thresh_batch * (1 - self.alpha) # moving average
        # self.thresh = 0.0 # without sample selection
        # self.thresh = thresh_batch # no moving average

        # if (scores >= 0.9).sum().item() == 0:
        #     pass
        # elif self.thresh < 0.9:
        #     self.thresh = 0.9 ### the minimum threshold is 0.9
        statistic = [scores.max().item(), scores.min().item(), scores.mean().item()]

        idx = scores >= self.thresh
        lbs = lbs[idx]

        G.load_state_dict(org_state_G)
        F.load_state_dict(org_state_F)
        if is_train:
            G.train()
            F.train()
        else:
            G.eval()
            F.eval()
        return lbs.detach(), idx, statistic


class LabelGuessor_dymaticT_with_pseudo_label(object):

    def __init__(self, alpha=0.9, init_thresh=1.0, mode='train'):
        self.alpha = alpha
        self.mode = mode
        self.thresh = init_thresh

    def __call__(self, G, F, ims, lbs, percent):
        # NOTE: lbs here is pseudo label
        org_state_G = {
            k: v.clone().detach()
            for k, v in G.state_dict().items()
        }
        org_state_F = {
            k: v.clone().detach()
            for k, v in F.state_dict().items()
        }
        is_train = G.training
        with torch.no_grad():
            if self.mode == 'train':
                G.train()
                F.train()
            elif self.mode == 'eval':
                G.eval()
                F.eval()

            logits = F(G(ims))
            probs = torch.softmax(logits, dim=1)

        scores = probs[torch.LongTensor(np.arange(probs.size(0))), lbs]
        sorted_values, _ = torch.sort(scores, descending=True)  ### from large to small

        num_batch = logits.size(0)
        selected = num_batch * percent
        if selected > num_batch:
            selected = num_batch
        round_number = round(selected)
        ceil_number = math.ceil(selected)
        ## get the threshold in this batch level
        if round_number == ceil_number:
            if round_number == 0:
                thresh_batch = 1.0
            else:
                thresh_batch = sorted_values[round_number - 1]
        else:
            ## round value >= ceil value
            if round_number == 0:
                round_value = 1.0
            else:
                round_value = sorted_values[round_number - 1]
            ceil_value = sorted_values[ceil_number - 1]
            ## get the thresh_value with interpolation
            thresh_batch = round_value - (selected - round_number) * (round_value - ceil_value)

        self.thresh = self.thresh * self.alpha + thresh_batch * (1 - self.alpha) # moving average
        # self.thresh = 0.0 # without sample selection
        # self.thresh = thresh_batch # no moving average

        # if (scores >= 0.9).sum().item() == 0:
        #     pass
        # elif self.thresh < 0.9:
        #     self.thresh = 0.9 ### the minimum threshold is 0.9
        statistic = [scores.max().item(), scores.min().item(), scores.mean().item()]

        idx = scores >= self.thresh
        lbs = lbs[idx]

        G.load_state_dict(org_state_G)
        F.load_state_dict(org_state_F)
        if is_train:
            G.train()
            F.train()
        else:
            G.eval()
            F.eval()
        return lbs.detach(), idx, statistic


class SampleFilter(object):
    r"""Filter out some samples of a mini batch
    Args:
        thresh_mode ("avg", "fix", "zero"): decide how the thresh is computed.
            avg: a moving average of thresh computed by each batch
            fix: compute thresh by current batch
            zero: use a 0.0 thresh, which means select all samples
        mode ("batch" | "full"): decide whether use score of a mini batch or
        the full batch to select samples
    """
    def __init__(self, alpha=0.9, init_thresh=0.0, thresh_mode="avg",
        mode="batch", order="descend"):
        # self.score_full_batch = score_full_batch
        self.alpha = alpha
        self.thresh = init_thresh
        self.thresh_mode = thresh_mode
        self.mode = mode
        if order == "ascend": ### from small to large
            self.descending = False
        elif order == "descend": ### from large to small
            self.descending = True
        else:
            raise ValueError("order should be 'ascend' or 'descend'")
        if mode == "full":
            raise ValueError("'full' sample_filter_mode is not supported in this version!")
            self.sorted_values, _ = torch.sort(score_full_batch, descending=self.descending)

    def __call__(self, scores, percent):
        if self.mode == "batch":
            sorted_values, _ = torch.sort(scores, descending=self.descending)
        elif self.mode == "full":
            sorted_values = self.sorted_values

        num_batch = sorted_values.size(0)
        selected = num_batch * percent
        if selected > num_batch:
            selected = num_batch
        round_number = round(selected)
        ceil_number = math.ceil(selected)

        ## get the threshold in this batch level
        if round_number == ceil_number:
            if round_number == 0:
                thresh_batch = 1.0
            else:
                thresh_batch = sorted_values[round_number - 1]
        else:
            ## round value >= ceil value
            if round_number == 0:
                round_value = 1.0
            else:
                round_value = sorted_values[round_number - 1]
            ceil_value = sorted_values[ceil_number - 1]
            ## get the thresh_value with interpolation
            thresh_batch = round_value - (selected - round_number) * (round_value - ceil_value)

        if self.thresh_mode == "avg":
            self.thresh = self.thresh * self.alpha + thresh_batch * (1 - self.alpha)  # moving average
        elif self.thresh_mode == "fix":
            self.thresh = thresh_batch  # no moving average
        elif self.thresh_mode == "zero":
            self.thresh = 0.0  # without sample selection
        else:
            raise NotImplementedError("mode: {}".format(self.thresh_mode))
        # print(thresh_batch, self.thresh)
        # if (scores >= 0.9).sum().item() == 0:
        #     pass
        # elif self.thresh < 0.9:
        #     self.thresh = 0.9 ### the minimum threshold is 0.9
        statistic = [scores.max().item(), scores.min().item(), scores.mean().item()]

        valid = scores >= self.thresh
        if valid.sum() <= 1:
            # avoid only one sample is selected, which will cause BatchNorm error
            valid_tmp = torch.zeros_like(valid).type(torch.BoolTensor)
            valid_tmp[scores.topk(2)[1]] = True
            valid = valid_tmp

        return valid, statistic


class CategoricalSampleFilter(object):
    r"""Filter out some samples of a mini batch
    Args:
        num_class (int): number of classes
        len_buffer (int): length of the class-wise buffer
    """
    def __init__(self, num_class, len_buffer=100):
        self.num_class = num_class
        self.len_buffer = len_buffer
        self.buffer_tensor = torch.zeros(self.num_class, self.len_buffer)
        self.thresh = 0  # useless, only for compatibility of a print sentence

    def __call__(self, scores, gt, percent):
        r"""
        Args:
            scores (torch.FloatTensor): its size is [batch_size]
            gt (torch.LongTensor): its size is [batch_size]
            percent (float): percent to select samples
        """
        ### check whether the scores satisfy the percent constrain (category wise)
        sorted_values, _ = torch.sort(self.buffer_tensor, 1, descending=True)
        pos_thr = int(self.len_buffer * percent) - 1  ## -1 ~ self.len_buffer-1
        if pos_thr < 0:
            pos_thr = 0
        elif pos_thr > self.len_buffer -1:
            pos_thr = self.len_buffer - 1
        prob_thr = sorted_values[:, pos_thr]  ## threshold for each category torch.size([self.num_class])

        valid = torch.zeros_like(scores, dtype=torch.bool) ## initialize valid as all False
        for i in range(self.num_class):
            valid_i = (gt == i) & (scores > prob_thr[i])
            valid = valid | valid_i

        ## push the current scores into buffer
        for i in range(self.num_class):
            score_category_i = scores[gt == i]
            len_i = score_category_i.size(0)
            if len_i == 0:
                continue
            elif len_i > self.len_buffer:
                len_i = self.len_buffer
                score_category_i = score_category_i[-len_i:]

            self.buffer_tensor[i][: -len_i] = self.buffer_tensor.clone()[i][len_i:]
            self.buffer_tensor[i][-len_i:] = score_category_i

        statistic = [scores.max().item(), scores.min().item(), scores.mean().item()]
        return valid, statistic


def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.momentum = 0.0  #### fix all the imagenet pre-trained running mean and average
        # m.eval()

def release_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.momentum = 0.1  #### roll back to the default setting
        # m.train()

def to_cuda(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return x

def to_cpu(x):
    return x.cpu()

def to_numpy(x):
    if torch.cuda.is_available():
        x = x.cpu()
    return x.data.numpy()

def to_onehot(label, num_classes):
    r"""
    Args:
        label (torch.Tensor): whose size is [1]
    Return:
        onehot (torch.Tensor): whose size is [1, num_classes]
    """    
    identity = torch.eye(num_classes).to(label.device)
    onehot = torch.index_select(identity, 0, label)
    return onehot


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.1)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.1)
        m.bias.data.fill_(0)


class histogram_collector():
    r"""Collect data and return a tensor for drawing a histogram"""
    def __init__(self, bins, min=0.0, max=1.0):
        self.bins = bins
        self.min = min
        self.max = max
        self.dis_output = torch.zeros(bins).cuda() # distribution of output

    def add_data(self, data):
        tmp = torch.histc(data.detach(), bins=self.bins, min=self.min, max=self.max)
        self.dis_output += tmp
    
    def compute(self, divisor=None):
        if divisor is None:
            divisor = self.dis_output.sum()
        self.dis_output /= divisor
        return self.dis_output.cpu().numpy()

    def clear_data(self):
        r"""Generally, if one calls compute(), then it has to follow a clear_data()"""
        self.dis_output.fill_(0.0)

class GifWriterManager():
    def __init__(self, save_dir, switches):
        # gif_writer is used to check the max score and accuracy at each max score region
        # gif_writer2 is used to observe the score of column of pesudo labels
        # gif_writer3 is used to observe the class-wise prediction and accuracy of model outputs
        self.filenames = ["score", "score_y", "class_wise_acc", "conf_acc", "inoutflow"]
        self.gif_writers = []
        self.save_dir = save_dir
        self.switches = switches

    def get_ready(self, loop):
        for filename, switch in zip(self.filenames, self.switches):
            if switch:
                self.gif_writers.append(imageio.get_writer(
                    os.path.join(self.save_dir, f"{filename}_loop{loop}.gif"), mode="I"))
            else:
                self.gif_writers.append(None)

    def reset(self):
        for writer, switch in zip(self.gif_writers, self.switches):
            if switch:
                writer.close()
        self.gif_writers.clear()

def plt2numpy(suffix):
    r"""Use a file suffix to avoid conflict during simultaneously running multi
    experiments.
    """
    fig_name = "/tmp/tmp{}.jpeg".format(suffix)
    plt.savefig(fig_name)
    img = Image.open(fig_name)
    image = np.asarray(img)

    return image


class ParamsComparator():
    def __init__(self):
        pass

    def set_start(self, model: nn.Module):
        self.model_begin = copy.deepcopy(model)
        # deepcopy will keep the device, so it needs manually set
        self.model_begin = self.model_begin.to(device="cpu")
        # remember to cancel gradient requirement
        for param in self.model_begin.parameters():
            param.requires_grad = False

    def compare(self, model: nn.Module):
        model_end = copy.deepcopy(model)
        model_end = model_end.to(device="cpu")
        for param in model_end.parameters():
            param.requires_grad = False

        result = []
        for (name, param), (name2, param2) in zip(self.model_begin.named_parameters(), model_end.named_parameters()):
            assert name == name2
            with torch.no_grad():
                tmp = param - param2
                result.append("{} max: {:+.5f} min: {:+.5f} mean: {:+.5f} abs mean: {:+.5f} size:{}\n".format(
                              name.ljust(45), tmp.max().item(), tmp.min().item(),
                              tmp.mean().item(), tmp.abs().mean().item(), list(tmp.size())))

        return result


class ParamsComparators():
    r"""Wrapper of ParamsComparator"""
    def __init__(self, namelist: List):
        self.namelist = namelist
        self.PCs = {}
        for name in namelist:
            self.PCs[name] = ParamsComparator()

    def set_start(self, model_dict: Dict):
        for name in self.namelist:
            self.PCs[name].set_start(model_dict[name])

    def compare(self, model_dict: Dict):
        result = {}
        for name in self.namelist:
            result[name] = self.PCs[name].compare(model_dict[name])
        return result


model_urls = {
    'resnet18': 'https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth',
    'resnet34': 'https://s3.amazonaws.com/pytorch/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth',
    'resnet101':
        'https://s3.amazonaws.com/pytorch/models/resnet101-5d3b4d8f.pth',
    'resnet152':
        'https://s3.amazonaws.com/pytorch/models/resnet152-b121ed2d.pth',
}


def accuracy_for_each_class(output, target, predict_vector, correct_vector):
    r"""Computes the precision for each class: n_correct_among_them / n_predict_x"""
    batch_size = target.size(0)
    _, pred = output.topk(1, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1)).float().cpu().squeeze()
    for i in range(batch_size):
        predict_vector[int(pred[0, i])] += 1
        correct_vector[int(pred[0, i])] += correct[i]

    return predict_vector, correct_vector


def accuracy_for_each_class_original(output, target, total_vector, correct_vector):
    r"""Computes the precision for each class: n_correct_among_them / n_truly_x"""
    batch_size = target.size(0)
    _, pred = output.topk(1, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1)).float().cpu().squeeze()
    for i in range(batch_size):
        total_vector[int(target[i])] += 1
        correct_vector[int(target[i])] += correct[i]

    return total_vector, correct_vector
