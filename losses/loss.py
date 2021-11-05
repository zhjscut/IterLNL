from typing import Optional

import torch
from torch import nn
from torch import Tensor
from torch.nn import Module
from torch.nn import _reduction as _Reduction
import torch.nn.functional as F


class _Loss(Module):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__()
        if size_average is not None or reduce is not None:
            self.reduction = _Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction


class _WeightedLoss(_Loss):
    def __init__(self, weight: Optional[Tensor] = None, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)
        self.register_buffer('weight', weight)


class GlobalDiverseLoss(_Loss):
    r"""Author: zhang.haojian
    This criterion aims to improve the diversity of output class, which is useful to 
    counter model whose output lack of some weak classes.

    The `input` is expected to contain raw, unnormalized scores for each class.

    `input` has to be a Tensor of size either :math:`(minibatch, C)` or
    :math:`(minibatch, C, d_1, d_2, ..., d_K)`
    with :math:`K \geq 1` for the `K`-dimensional case (described later).

    This criterion expects a class index in the range :math:`[0, C-1]` as the
    `target` for each value of a 1D tensor of size `minibatch`; if `ignore_index`
    is specified, this criterion also accepts this class index (this index may not
    necessarily be in the class range).

    The loss can be described as:

    .. math::
        \text{loss}(input) = \sum_{k=1}^K \bar p_k \log \bar p_k, where
        \bar p = E_{x_i\in X}[\text{Softmax}(input_i)]

    The losses are averaged across observations for each minibatch.

    Args:
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``

    Shape:
        - Input: :math:`(N, C)` where `C = number of classes`, or
          :math:`(N, C, d_1, d_2, ..., d_K)` with :math:`K \geq 1`
          in the case of `K`-dimensional loss.
        - Target: :math:`(N)` where each value is :math:`0 \leq \text{targets}[i] \leq C-1`, or
          :math:`(N, d_1, d_2, ..., d_K)` with :math:`K \geq 1` in the case of
          K-dimensional loss.
        - Output: scalar.
          If :attr:`reduction` is ``'none'``, then the same size as the target:
          :math:`(N)`, or
          :math:`(N, d_1, d_2, ..., d_K)` with :math:`K \geq 1` in the case
          of K-dimensional loss.

    Examples::

        >>> loss = GlobalDiverseLoss()
        >>> input = torch.randn(3, 5, requires_grad=True)
        >>> output = loss(input)
        >>> output.backward()
    """
    __constants__ = ['reduction']

    def __init__(self, reduction: str = 'mean') -> None:
        super().__init__(reduction=reduction)

    def forward(self, input: Tensor) -> Tensor:
        soft = nn.Softmax(dim=1)(input)
        mean_soft = soft.mean(dim=0)
        return torch.sum(-mean_soft * torch.log(mean_soft + 1e-5))


class DistillationLoss(_Loss):
    r"""Author: zhang.haojian
    This criterion aims to make model's output similar to another model.

    The `input` is expected to contain raw, unnormalized scores for each class.

    `input1` has to be a Tensor of size either :math:`(minibatch, C)` or
    :math:`(minibatch, C, d_1, d_2, ..., d_K)`
    with :math:`K \geq 1` for the `K`-dimensional case (described later).

    `input2` is the same as `input1` in format.

    The loss can be described as:

    .. math::
        \text{loss}(input) = \sum_{k=1}^K p_k \log q_k, where
        p_i = \text{Softmax}(f_1(data_i))
        q_i = f_2(data_i)

    The losses are averaged across observations for each minibatch.

    Args:
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``

    Shape:
        - Prior: :math:`(N, C)` where `C = number of classes`, or
          :math:`(N, C, d_1, d_2, ..., d_K)` with :math:`K \geq 1`
          in the case of `K`-dimensional loss.
        - Input: :math:`(N, C)` where `C = number of classes`, or
          :math:`(N, C, d_1, d_2, ..., d_K)` with :math:`K \geq 1`
          in the case of `K`-dimensional loss.          
        - Output: scalar.
          If :attr:`reduction` is ``'none'``, then the same size as the target:
          :math:`(N)`, or
          :math:`(N, d_1, d_2, ..., d_K)` with :math:`K \geq 1` in the case
          of K-dimensional loss.

    Examples:

        >>> loss = DistillationLoss()
        >>> prior = torch.randn(3, 5)
        >>> input = torch.randn(3, 5, requires_grad=True)
        >>> output = loss(prior, input)
        >>> output.backward()
    """
    __constants__ = ['reduction']

    def __init__(self, reduction: str = 'mean') -> None:
        super().__init__(reduction=reduction)

    def forward(self, prior: Tensor, input: Tensor) -> Tensor:
        input = nn.Softmax(dim=1)(input)
        # it is said that put the tensor that need update into log() is easier to be optimized
        return torch.sum(-prior * torch.log(input + 1e-5), dim=1).mean()


class ZeroOneLoss():
    r"""Author: zhang.haojian
    This criterion is used to compute 0-1 loss, sometimes used in early stopping by computing
    whether the prediction in adjacent epochs is close enough.
    Note that it does not support gradient back propagation.

    Shape:
        - Input1: :math:`(N)`
        - Input2: :math:`(N)`
        
    Example:
        >>> loss = ZeroOneLoss()
        >>> input1 = torch.tensor([1,2,3,4,5], dtype=torch.int64)
        >>> input2 = torch.tensor([3,2,4,4,5], dtype=torch.int64)
        >>> output = loss(input1, input2)
    """
    def __init__(self) -> None:
        pass

    def __call__(self, input1: Tensor, input2: Tensor) -> Tensor:
        # both input and target is a label vector
        return float((input1 != input2).sum()) / input1.size(0)


class SmoothCELoss(_Loss):
    r"""Author: zhang.haojian
    This criterion is used in label smoothing trick, calculating cross entropy loss between 
    output logits and a soft label [1-δ, δ/(C-1), ..., δ/(C-1)].

    The `input` is expected to contain raw, unnormalized scores for each class.

    `input` has to be a Tensor of size either :math:`(minibatch, C)` or
    :math:`(minibatch, C, d_1, d_2, ..., d_K)`
    with :math:`K \geq 1` for the `K`-dimensional case (described later).

    This criterion expects a class index in the range :math:`[0, C-1]` as the
    `target` for each value of a 1D tensor of size `minibatch`; if `ignore_index`
    is specified, this criterion also accepts this class index (this index may not
    necessarily be in the class range).

    The loss can be described as:

    .. math::
        \text{loss}(x, class) = -\sum_i q[i] \log\left(\frac{\exp(x[i])}{\sum_j \exp(x[j])}\right), where
        q[class] = 1 - \delta, q[c] = \delta / (C - 1), \forall c \neq class

    The losses are averaged across observations for each minibatch.

    Args:
        delta (float): \delta in label smoothing equation
        num_classes (int): number of classes
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``

    Shape:
        - Input: :math:`(N, C)` where `C = number of classes`, or
          :math:`(N, C, d_1, d_2, ..., d_K)` with :math:`K \geq 1`
          in the case of `K`-dimensional loss.
        - Target: :math:`(N)` where each value is :math:`0 \leq \text{targets}[i] \leq C-1`, or
          :math:`(N, d_1, d_2, ..., d_K)` with :math:`K \geq 1` in the case of
          K-dimensional loss.
        - Output: scalar.
          If :attr:`reduction` is ``'none'``, then the same size as the target:
          :math:`(N)`, or
          :math:`(N, d_1, d_2, ..., d_K)` with :math:`K \geq 1` in the case
          of K-dimensional loss.

    Examples::
        >>> loss = SmoothCEloss()
        >>> input = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.empty(3, dtype=torch.long).random_(5)
        >>> output = loss(input, target)
        >>> output.backward()
    """
    __constants__ = ['reduction']

    def __init__(self, delta, num_classes, reduction: str = 'mean') -> None:
        super().__init__(reduction=reduction)
        self.delta = delta
        self.C = num_classes

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # modified from https://blog.csdn.net/qq_36560894/article/details/118424356
        # NOTE: at present this loss do no reduction in distributed training
        logprobs = F.log_softmax(input, dim=1)	# softmax + log
        target = F.one_hot(target, self.C)	# transfrom single int to one-hot vector
        target = torch.clamp(target.float(), min=self.delta / (self.C - 1), max=1.0 - self.delta)
        return -torch.sum(target * logprobs, 1).mean()
