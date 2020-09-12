import torch
from torch import nn
import torch.nn.functional as functional
from torch.autograd import Function
import numpy as np

import sys
sys.path.append("../../")
from modules.deq import *


class TrellisNetDEQModule(DEQModule):

    """ See DEQModule class for documentation """

    def __init__(self, func, func_copy):
        super(TrellisNetDEQModule, self).__init__(func, func_copy)
        
    def _solve_by_subseq(self, z1s, us, z0, pos_emb, threshold, train_step, subseq_len=100):
        z1s_out = torch.zeros_like(z1s)
        with torch.no_grad():
            z0_temp = z0
            for t in range(0, seq_len, subseq_len):
                z1ss = z1s[:,:,t:t+subseq_len]
                uss = us[:,:,t:t+subseq_len]    # Different from transformer, in TrellisNet, inject only on current seq.

                # Invoke DEQ forward module. args = [threshold, train_step]
                res = RootFind.apply(self.func, z1ss, uss, z0_temp, threshold, train_step)
                z0_temp = res[:,:,-1:]
                z1s_out[:,:,t:t+subseq_len] = res
        
        z1s = z1s_out
        z1s_out = torch.zeros_like(z1s)
        z0_temp = z0
        for t in range(0, seq_len, subseq_len):
            z1ss = z1s[:, :, t:t + subseq_len]
            uss = us[:, :, t:t + subseq_len]

            new_z1ss = RootFind.f(self.func, z1ss, uss, z0_temp, threshold, train_step)
            res = self.Backward.apply(self.func_copy, new_z1ss, uss, z0_temp, threshold, train_step)
            z0_temp = res[:,:,-1:]
            z1s_out[:,:,t:t+subseq_len] = res
        return z1s_out

    def forward(self, z1s, us, z0, **kwargs):
        bsz, total_hsize, seq_len = z1s.size()
        train_step = kwargs.get('train_step', -1)
        subseq_len = kwargs.get('subseq_len', seq_len)
        threshold = kwargs.get('threshold', 50)

        if us is None:
            raise ValueError("Input injection is required.")

        # Use this line for longer sequences: 
        #     self._solve_by_subseq(z1s, us, z0, threshold, train_step, subseq_len=subseq_len)

        # Use these lines for shorter sequences:
        z1s_out = RootFind.apply(self.func, z1s, us, z0, threshold, train_step)
        if self.training:
            z1s_out = RootFind.f(self.func, z1s_out, us, z0, threshold, train_step)
            z1s_out = self.Backward.apply(self.func_copy, z1s_out, us, z0, threshold, train_step)
        return z1s_out
