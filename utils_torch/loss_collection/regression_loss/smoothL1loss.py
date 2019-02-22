import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

# __all__ = ['WeightSmoothL1Loss', 'FocalLoss', 'MultiTaskLoss', 'FocalSoftmax']


class WeightSmoothL1Loss(nn.SmoothL1Loss):
    def __init__(self, size_average=True, reduce=True):
        super(WeightSmoothL1Loss, self).__init__()
        self.size_average = size_average
        self.reduce = reduce

    def forward(self, inputs, targets, w_in, w_out, sigma=10., drop_thres=None, OHEM_thres=None):  # sigma=10 or 1?
        # print inputs.size()
        targets.requires_grad = False
        w_in.requires_grad = False
        w_out.requires_grad = False
        sigma2 = sigma * sigma
        diff = (inputs - targets) * w_in

        # drop abs(error) < 0.01
        if drop_thres is not None:
            diff_lt_thres = diff.abs().ge(drop_thres).float()
            diff = diff * diff_lt_thres

        diff_masked = torch.masked_select(diff, w_in.byte())
        abs_diff_masked = diff_masked.abs()

        if OHEM_thres is not None:
            top_k_value, top_k_idx = torch.topk(abs_diff_masked, int(OHEM_thres * abs_diff_masked.size(0)))

        if w_in.sum() != 0:
            abs_diff_std = abs_diff_masked.var()
            abs_diff_mean = abs_diff_masked.mean()
        else:
            abs_diff_std = 0
            abs_diff_mean = 0

        output = abs_diff_masked.lt(1.0 / sigma2).float() * torch.pow(diff_masked, 2) * \
                 0.5 * sigma2 + abs_diff_masked.ge(1.0 / sigma2).float() * (abs_diff_masked - 0.5 / sigma2)

        # output = output*w_out
        if self.reduce:
            if self.size_average:
                # return output.mean(), abs_diff_mean, abs_diff_std
                # print 'using size average'
                return output.sum(), abs_diff_mean, abs_diff_std, diff_masked
            else:
                return output.sum() / inputs.size(0), abs_diff_mean, abs_diff_std, diff_masked
        else:
            return output.sum(1).sum(1), abs_diff_mean, abs_diff_std, diff_masked


class WeightSmoothL1Lossbackup(nn.SmoothL1Loss):
    def __init__(self, size_average=True, reduce=True):
        super(WeightSmoothL1Loss, self).__init__()
        self.size_average = size_average
        self.reduce = reduce

    def forward(self, inputs, targets, w_in, w_out, sigma=10.):  # sigma=10 or 1?
        # print inputs.size()
        targets.requires_grad = False
        w_in.requires_grad = False
        w_out.requires_grad = False
        sigma2 = sigma * sigma
        diff = (inputs - targets) * w_in
        abs_diff = diff.abs()

        diff_masked = torch.masked_select(diff, w_in.byte())
        # diff_masked = []
        # abs_diff_masked = torch.masked_select(abs_diff, w_in.byte())
        abs_diff_std = diff_masked.abs().var()
        abs_diff_mean = diff_masked.abs().mean()

        output = abs_diff.lt(1.0 / sigma2).float() * torch.pow(diff, 2) * \
                 0.5 * sigma2 + abs_diff.ge(1.0 / sigma2).float() * (abs_diff - 0.5 / sigma2)
        output = output * w_out
        if self.reduce:
            if self.size_average:
                # return output.mean(), abs_diff_mean, abs_diff_std
                # print 'using size average'
                return output.sum(), abs_diff_mean, abs_diff_std, diff_masked
            else:
                return output.sum() / inputs.size(0), abs_diff_mean, abs_diff_std, diff_masked
        else:
            return output.sum(1).sum(1), abs_diff_mean, abs_diff_std, diff_masked


def smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, sigma=1.0, dim=[1]):
    """
    :param bbox_pred:
    :param bbox_targets:
    :param bbox_inside_weights:
    :param bbox_outside_weights:
    :param sigma:  控制分段函数的分段文字　-1/sigma_2   1/sigma_2
    :param dim:
    :return:
    """
    sigma_2 = sigma ** 2
    box_diff = bbox_pred - bbox_targets
    in_box_diff = bbox_inside_weights * box_diff
    abs_in_box_diff = torch.abs(in_box_diff)
    smoothL1_sign = (abs_in_box_diff < 1. / sigma_2).detach().float()
    in_loss_box = torch.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
                  + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
    out_loss_box = bbox_outside_weights * in_loss_box
    loss_box = out_loss_box
    for i in sorted(dim, reverse=True):
        loss_box = loss_box.sum(i)
    loss_box = loss_box.mean()
    return loss_box


class BiasFocalLoss(nn.Module):
    def __init__(self, alpha=0.800000011921, gamma=2.0, bias=0, size_average=True, reduce=True):
        super(BiasFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.size_average = size_average
        self.reduce = reduce
        self.bias = bias

    def forward(self, inputs, targets):
        # print targets

        p = inputs.sigmoid() + self.bias
        term1 = torch.pow(1-p, self.gamma)*(p.log())
        term2 = torch.pow(p, self.gamma)*(-inputs*inputs.ge(0).float() -
                                          (1+(inputs-2*inputs*inputs.ge(0).float()).exp()).log())
        loss = - self.alpha * \
            targets.eq(1).float()*term1 - (1-self.alpha) * \
            targets.eq(0).float()*term2

        if self.reduce:
            if self.size_average:
                return loss.sum()/targets.ge(0).float().sum()
            else:
                return loss.sum()/targets.size(0)
        else:
            return loss.view(loss.size(0), -1).sum(1).unsqueeze(1)


if __name__=='__main__':

    for i in range(5):
        input = torch.autograd.Variable(torch.randn(80, 80, 80))
        target = torch.autograd.Variable(torch.randn(80, 80, 80))
        loss = smooth_l1_loss(input, target, 0.3, 0.6)
        # loss = WeightSmoothL1Loss()
        # print(output.data)




