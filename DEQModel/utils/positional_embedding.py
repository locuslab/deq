import torch
from torch import nn


class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(self.inv_freq, pos_seq)    # C x L
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=0)   # Concat at feature dimension

        if bsz is not None:
            return pos_emb[None,:,:].expand(bsz, -1, -1)
        else:
            return pos_emb[None,:,:]