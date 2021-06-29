import torch
import torch.nn.functional as F
import torch.nn as nn


def list2vec(z1_list):
    """Convert list of tensors to a vector"""
    bsz = z1_list[0].size(0)
    return torch.cat([elem.reshape(bsz, -1, 1) for elem in z1_list], dim=1)


def vec2list(z1, cutoffs):
    """Convert a vector back to a list, via the cutoffs specified"""
    bsz = z1.shape[0]
    z1_list = []
    start_idx, end_idx = 0, cutoffs[0][0] * cutoffs[0][1] * cutoffs[0][2]
    for i in range(len(cutoffs)):
        z1_list.append(z1[:, start_idx:end_idx].view(bsz, *cutoffs[i]))
        if i < len(cutoffs)-1:
            start_idx = end_idx
            end_idx += cutoffs[i + 1][0] * cutoffs[i + 1][1] * cutoffs[i + 1][2]
    return z1_list


def conv3x3(in_planes, out_planes, stride=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=bias)

def conv5x5(in_planes, out_planes, stride=1, bias=False):
    """5x5 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=stride, padding=2, bias=bias)


def norm_diff(new, old, show_list=False):
    if show_list:
        return [(new[i] - old[i]).norm().item() for i in range(len(new))]
    return np.sqrt(sum((new[i] - old[i]).norm().item()**2 for i in range(len(new))))