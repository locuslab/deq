from torch.nn.parameter import Parameter
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable

##############################################################################################################
#
# Temporal DropConnect in a feed-forward setting
#
##############################################################################################################

class WeightDrop(torch.nn.Module):
    def __init__(self, module, weights, dropout=0, temporal=True):
        """
        Weight DropConnect, adapted from a recurrent setting by Merity et al. 2017

        :param module: The module whose weights are to be applied dropout on
        :param weights: A 2D list identifying the weights to be regularized. Each element of weights should be a
                        list containing the "path" to the weight kernel. For instance, if we want to regularize
                        module.layer2.weight3, then this should be ["layer2", "weight3"].
        :param dropout: The dropout rate (0 means no dropout)
        :param temporal: Whether we apply DropConnect only to the temporal parts of the weight (empirically we found
                         this not very important)
        """
        super(WeightDrop, self).__init__()
        self.module = module
        self.weights = weights
        self.dropout = dropout
        self.temporal = temporal
        if self.dropout > 0.0:
            self._setup()

    def _setup(self):
        for path in self.weights:
            full_name_w = '.'.join(path)

            module = self.module
            name_w = path[-1]
            for i in range(len(path) - 1):
                module = getattr(module, path[i])
            w = getattr(module, name_w)
            del module._parameters[name_w]
            module.register_parameter(name_w + '_raw', Parameter(w.data))

    def _setweights(self):
        for path in self.weights:
            module = self.module
            name_w = path[-1]
            for i in range(len(path) - 1):
                module = getattr(module, path[i])
            raw_w = getattr(module, name_w + '_raw')

            if len(raw_w.size()) > 2 and raw_w.size(2) > 1 and self.temporal:
                # Drop the temporal parts of the weight; if 1x1 convolution then drop the whole kernel
                w = torch.cat([F.dropout(raw_w[:, :, :-1], p=self.dropout, training=self.training),
                               raw_w[:, :, -1:]], dim=2)
            else:
                w = F.dropout(raw_w, p=self.dropout, training=self.training)

            setattr(module, name_w, w)

    def forward(self, *args, **kwargs):
        if self.dropout > 0.0:
            self._setweights()
        return self.module.forward(*args, **kwargs)


def matrix_diag(a, dim=2):
    """
    a has dimension (N, (L,) C), we want a matrix/batch diag that produces (N, (L,) C, C) from the last dimension of a
    """
    if dim == 2:
        res = torch.zeros(a.size(0), a.size(1), a.size(1))
        res.as_strided(a.size(), [res.stride(0), res.size(2)+1]).copy_(a)
    else:
        res = torch.zeros(a.size(0), a.size(1), a.size(2), a.size(2))
        res.as_strided(a.size(), [res.stride(0), res.stride(1), res.size(3)+1]).copy_(a)
    return res

##############################################################################################################
#
# Embedding dropout
#
##############################################################################################################

def embedded_dropout(embed, words, dropout=0.1, scale=None):
    """
    Apply embedding encoder (whose weight we apply a dropout)

    :param embed: The embedding layer
    :param words: The input sequence
    :param dropout: The embedding weight dropout rate
    :param scale: Scaling factor for the dropped embedding weight
    :return: The embedding output
    """
    if dropout:
        mask = embed.weight.data.new().resize_((embed.weight.size(0), 1)).bernoulli_(1 - dropout).expand_as(
            embed.weight) / (1 - dropout)
        mask = Variable(mask)
        masked_embed_weight = mask * embed.weight
    else:
        masked_embed_weight = embed.weight

    if scale:
        masked_embed_weight = scale.expand_as(masked_embed_weight) * masked_embed_weight

    padding_idx = embed.padding_idx
    if padding_idx is None:
        padding_idx = -1

    X = F.embedding(words, masked_embed_weight, padding_idx, embed.max_norm, embed.norm_type,
                                       embed.scale_grad_by_freq, embed.sparse)
    return X



##############################################################################################################
#
# Variational dropout (for input/output layers, and for hidden layers)
#
##############################################################################################################

class VariationalDropout(nn.Module):
    def __init__(self):
        """
        Feed-forward version of variational dropout that applies the same mask at every time step
        """
        super(VariationalDropout, self).__init__()

    def forward(self, x, dropout=0.5, dim=3):
        if not self.training or not dropout:
            return x
        if dim == 4:
            # Dimension (M, N, L, C), where C stands for channels
            m = torch.zeros_like(x[:,:,:1]).bernoulli_(1 - dropout)
        else:
            # Dimension (N, L, C)
            m = torch.zeros_like(x[:,:1]).bernoulli_(1 - dropout)
        mask = m.requires_grad_(False) / (1 - dropout)
        mask = mask.expand_as(x).to(x)
        return mask * x


class VariationalHidDropout(nn.Module):
    def __init__(self, dropout=0.0, length_first=False):
        """
        Hidden-to-hidden (VD-based) dropout that applies the same mask at every time step and every layer
        :param dropout: The dropout rate (0 means no dropout is applied)
        :param temporal: Whether the dropout mask is the same across the temporal dimension (or only the depth dimension)
        """
        super(VariationalHidDropout, self).__init__()
        self.dropout = dropout
        self.mask = None
        self.length_first = length_first

    def reset_mask(self, bsz, d, length):
        if self.length_first:
            # Dimension (N, L, C)
            m = torch.zeros(bsz, 1, d).bernoulli_(1 - self.dropout)
        else:
            # Dimension (N, C, L)
            m = torch.zeros(bsz, d, 1).bernoulli_(1 - self.dropout)
        mask = m.requires_grad_(False) / (1 - self.dropout)
        self.mask = mask
        return mask

    def forward(self, x):
        if not self.training or self.dropout == 0:
            return x
        assert self.mask is not None, f"You need to reset mask before using {self.__class__.__name__}"
        mask = self.mask.expand_as(x)  # Make sure the dimension matches
        return mask * x
    
    
class VariationalAttnDropout(VariationalHidDropout):
    def __init__(self, dropout=0.0, temporal=True):
        super(VariationalAttnDropout, self).__init__(dropout)

    def reset_mask(self, bsz, n_head, qlen, klen):
        # Dimension (N, n_head, L1, L2)
        m = torch.zeros(bsz, n_head, qlen, klen).bernoulli_(1 - self.dropout)
        mask = m.requires_grad_(False) / (1 - self.dropout)
        self.mask = mask
        return mask


class VariationalHidDropout2d(VariationalHidDropout):
    def __init__(self, dropout=0.0, spatial=True):
        """
        Hidden-to-hidden (VD-based) 2D dropout that applies the same mask at every layer
        :param spatial: If True, then all pixels of the HxW feature map will be applied the
                        same mask as well (i.e., certain entire channels of all pixels may be 
                        masked out).
        """
        super(VariationalHidDropout2d, self).__init__(dropout)
        self.spatial = spatial

    def reset_mask(self, bsz, d, H, W):
        # Dimension (N, C, H, W)
        if self.spatial:
            m = torch.zeros(bsz, d, 1, 1).bernoulli_(1 - self.dropout)
        else:
            m = torch.zeros(bsz, d, H, W).bernoulli_(1 - self.dropout)
        mask = m.requires_grad_(False) / (1 - self.dropout)
        self.mask = mask
        return mask

##############################################################################################################
#
# Weight normalization. Modified from the original PyTorch's implementation of weight normalization.
#
##############################################################################################################

def _norm(p, dim):
    """Computes the norm over all dimensions except dim"""
    if dim is None:
        return p.norm()
    elif dim == 0:
        output_size = (p.size(0),) + (1,) * (p.dim() - 1)
        return p.contiguous().view(p.size(0), -1).norm(dim=1).view(*output_size)
    elif dim == p.dim() - 1:
        output_size = (1,) * (p.dim() - 1) + (p.size(-1),)
        return p.contiguous().view(-1, p.size(-1)).norm(dim=0).view(*output_size)
    else:
        return _norm(p.transpose(0, dim), 0).transpose(0, dim)


class WeightNorm(object):
    def __init__(self, names, dim):
        """
        Weight normalization module

        :param names: The list of weight names to apply weightnorm on
        :param dim: The dimension of the weights to be normalized
        """
        self.names = names
        self.dim = dim

    def compute_weight(self, module, name):
        g = getattr(module, name + '_g')
        v = getattr(module, name + '_v')
        return v * (g / _norm(v, self.dim))

    @staticmethod
    def apply(module, names, dim):
        fn = WeightNorm(names, dim)

        for name in names:
            weight = getattr(module, name)

            # remove w from parameter list
            del module._parameters[name]

            # add g and v as new parameters and express w as g/||v|| * v
            module.register_parameter(name + '_g', Parameter(_norm(weight, dim).data))
            module.register_parameter(name + '_v', Parameter(weight.data))
            setattr(module, name, fn.compute_weight(module, name))

        # recompute weight before every forward()
        module.register_forward_pre_hook(fn)
        return fn

    def remove(self, module):
        for name in self.names:
            weight = self.compute_weight(module, name)
            delattr(module, name)
            del module._parameters[name + '_g']
            del module._parameters[name + '_v']
            module.register_parameter(name, Parameter(weight.data))

    def reset(self, module):
        for name in self.names:
            setattr(module, name, self.compute_weight(module, name))

    def __call__(self, module, inputs):
        # Typically, every time the module is called we need to recompute the weight. However,
        # in the case of TrellisNet, the same weight is shared across layers, and we can save
        # a lot of intermediate memory by just recomputing once (at the beginning of first call).
        pass


def weight_norm(module, names, dim=0):
    fn = WeightNorm.apply(module, names, dim)
    return module, fn
