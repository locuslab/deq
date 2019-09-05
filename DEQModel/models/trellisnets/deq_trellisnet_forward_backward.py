import torch
from torch import nn
import torch.nn.functional as functional
from torch.autograd import Function
import numpy as np

import sys
sys.path.append("../../")
from modules.deq import *


class TrellisNetDEQForward(DEQForward):

    """ See DEQForward class for documentation """

    def __init__(self, func):
        super(TrellisNetDEQForward, self).__init__(func)

    def _solve_equi(self, z1s, us, z0, **kwargs):
        bsz, total_hsize, seq_len = z1s.size()
        train_step = kwargs.get('train_step', -1)
        subseq_len = kwargs.get('subseq_len', seq_len)
        threshold = kwargs.get('threshold', 50)

        if us is None:
            raise ValueError("Input injection is required.")

        z1s_out = torch.zeros_like(z1s)
        with torch.no_grad():
            z0_temp = z0
            for t in range(0, seq_len, subseq_len):
                z1ss = z1s[:,:,t:t+subseq_len]
                uss = us[:,:,t:t+subseq_len]    # Different from transformer, in TrellisNet, inject only on current seq.

                # Invoke DEQ forward module. args = [threshold, train_step]
                res = DEQFunc.apply(self.func, z1ss, uss, z0_temp, threshold, train_step)
                z0_temp = res[:,:,-1:]
                z1s_out[:,:,t:t+subseq_len] = res
            return z1s_out


class TrellisNetDEQBackward(DEQBackward):

    """ See DEQBackward class for documentation """

    def __init__(self, func, func_copy):
        super(TrellisNetDEQBackward, self).__init__(func, func_copy)

    def _solve_back(self, z1s, us, z0, **kwargs):
        bsz, total_hsize, seq_len = z1s.size()
        train_step = kwargs.get('train_step', -1)
        subseq_len = kwargs.get('subseq_len', seq_len)
        threshold = kwargs.get('threshold', 80)

        if us is None:
            raise ValueError("Input injection is required.")

        z1s_out = torch.zeros_like(z1s)
        z0_temp = z0
        for t in range(0, seq_len, subseq_len):
            z1ss = z1s[:, :, t:t + subseq_len]
            uss = us[:, :, t:t + subseq_len]

            new_z1ss = DEQFunc.f(self.func, z1ss, uss, z0_temp, threshold, train_step)
            res = DummyDEQFunc.apply(self.func_copy, new_z1ss, uss, z0_temp, threshold, train_step)
            z0_temp = res[:,:,-1:]
            z1s_out[:,:,t:t+subseq_len] = res
        return z1s_out