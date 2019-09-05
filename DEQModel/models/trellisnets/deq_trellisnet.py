import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Function
import sys
import copy

sys.path.append('../../')
from modules.optimizations import *
from models.trellisnets.deq_trellisnet_forward_backward import TrellisNetDEQForward, TrellisNetDEQBackward
from utils.proj_adaptive_softmax import ProjectedAdaptiveLogSoftmax

__author__ = "shaojieb"
    

class MixSoftmax(nn.Module):
    def __init__(self, n_components, n_classes, nlasthid, ninp, decoder, dropoutl):
        """
        Apply mixture of softmax to the last layer of the network

        :param n_components: The number of softmaxes to use
        :param n_classes: The number of classes to predict
        :param nlasthid: The dimension of the last hidden layer from the deep network
        :param ninp: The embedding size
        :param decoder: The decoder layer
        :param dropoutl: The dropout to be applied on the pre-softmax output
        """
        super(MixSoftmax, self).__init__()
        self.n_components = n_components
        self.n_classes = n_classes
        self.prior = nn.Linear(nlasthid, n_components)          # C ---> m
        self.latent = nn.Linear(nlasthid, n_components * ninp)  # C ---> m*C
        self.decoder = decoder
        self.var_drop = VariationalDropout()
        self.ninp = ninp
        self.nlasthid = nlasthid
        self.dropoutl = dropoutl

    def init_weights(self):
        initrange = 0.1
        self.prior.weight.data.uniform_(-initrange, initrange)
        self.latent.weight.data.uniform_(-initrange, initrange)

    def forward(self, context):
        n_components = self.n_components
        n_classes = self.n_classes
        decoder = self.decoder
        ninp = self.ninp

        batch_size, seq_len, _ = context.size()
        priors = F.softmax(self.prior(context), dim=2).view(-1, n_components)            # ((bsz * seq_len) x m)
        latent = self.var_drop(self.latent(context), self.dropoutl)
        latent = F.softmax(decoder(F.tanh(latent.view(-1, n_components, ninp))), dim=2)  # ((bsz * seq_len) x m x n_classes)
        return (priors.unsqueeze(2).expand_as(latent) * latent).sum(1).view(batch_size, seq_len, n_classes)


class WeightShareConv1d(nn.Module):
    def __init__(self, n_hid, n_out, kernel_size, dropouth=0.0):
        """
        The weight-tied 1D convolution used in TrellisNet.
        :param n_hid: The dim of hidden input
        :param n_out: The dim of the pre-activation (i.e. convolutional) output
        :param kernel_size: The size of the convolutional kernel
        :param dropouth: Hidden-to-hidden dropout
        """
        super(WeightShareConv1d, self).__init__()
        self.kernel_size = kernel_size

        conv = nn.Conv1d(n_hid, n_out, kernel_size)
        self.weight = conv.weight
        self.bias = conv.bias
        self.init_weights()

        self.dict = dict()
        self.drop = VariationalHidDropout(dropout=dropouth)

    def init_weights(self):
        bound = 0.01
        self.weight.data.normal_(0, bound)
        self.bias.data.normal_(0, bound)

    def copy(self, func):
        self.weight.data = func.weight.data.clone().detach()
        self.bias.data = func.bias.data.clone().detach()
        self.drop.mask = func.drop.mask.clone().detach()

    def forward(self, x, dilation=1, hid=None):
        k = self.kernel_size
        padding = (k - 1) * dilation    # To maintain causality constraint
        x = F.pad(x, (padding, 0))

        # Hidden part
        if hid is not None:
            x[:,:,:padding] = hid.repeat(1, 1, padding)       # Note: we only pad the hidden part :-)
        res = F.conv1d(self.drop(x), self.weight, self.bias, dilation=dilation)
        return res


class TrellisNetLayer(nn.Module):
    def __init__(self, ninp, nhid, nout, kernel_size=2, dropouth=0.0, dilation=1):
        super().__init__()
        self.ninp = ninp
        self.nhid = nhid
        self.nout = nout
        self.kernel_size = kernel_size
        self.h_size = h_size = nhid + nout
        self.total_hsize = h_size
        self.intermediate_hsize = h_size
        self.inject_hsize = h_size
        self.dilation = dilation
        self.dropouth = dropouth

    def setup(self):
        h_size, intermediate_hsize = self.h_size, self.intermediate_hsize
        kernel_size, dropouth = self.kernel_size, self.dropouth
        self.full_conv = WeightShareConv1d(h_size, intermediate_hsize, kernel_size=kernel_size, dropouth=dropouth)

    def wnorm(self):
        self.full_conv, self.fn = weight_norm(module=self.full_conv, names=['weight'], dim=0)

    def copy(self, func):
        """
        Copy the parameters of func to self (all parameters are in the full_conv module). Note that the `weight_g`
        and `weight_v` parameters in weight normalization are NOT copied, because we only need the weight for
        func_copy.

        :param func: The original module to make copy of (must be of type `torch.nn.Module`)
        :return: None
        """
        self.full_conv.copy(func.full_conv)

    def reset(self, bsz, seq_len):
        # Recompute weight normalization weights
        if 'fn' in self.__dict__:
            self.fn.reset(self.full_conv)

        # Reset the variational dropout mask of the full_conv module
        self.full_conv.drop.reset_mask(torch.zeros(bsz, self.h_size, seq_len))

    def forward(self, z1ss, uss, z0, *args):
        """
        The transformation f_\theta in TrellisNet

        :param z1ss: Dimension (bsz x (*)d_model x seq_len)
        :param uss: Dimension (bsz x (*)d_model x seq_len)
        :param z0: Dimension (bsz x (*)d_model x seq_len)
        :param args: Extra arguments in a list
        :return:
        """
        raise NotImplementedError


class TrellisNetLSTMLayer(TrellisNetLayer):
    def __init__(self, ninp, nhid, nout, kernel_size=2, dropouth=0.0, dilation=1):
        super(TrellisNetLSTMLayer, self).__init__(ninp, nhid, nout, kernel_size, dropouth, dilation)
        self.total_hsize = 2 * self.h_size
        self.intermediate_hsize = 4 * self.h_size
        self.inject_hsize = 4 * self.h_size
        self.setup()

    def forward(self, z1ss, uss, z0, *args):
        """
        The transformation f_\theta in TrellisNetLSTM

        :param z1ss: Dimension (bsz x (2*d_model) x seq_len)
        :param uss: Dimension (bsz x (4*d_model) x seq_len)
        :param z0: Dimension (bsz x (2*d_model) x 1)
        :param args: Extra arguments in a list
        :return:
        """
        z1h, z1c = z1ss.chunk(2, dim=1)
        z0h, z0c = z0.chunk(2, dim=1)
        z0c = torch.cat([z0c, z1c], dim=2)[:,:,:-1]   # bsz x d_model x seq_len

        out = uss + self.full_conv(z1h, dilation=self.dilation, hid=z0h)
        it, ot, gt, ft = out.chunk(4, dim=1)
        it, ot, gt, ft = torch.sigmoid(it), torch.sigmoid(ot), torch.tanh(gt), torch.sigmoid(ft)
        ct = ft * z0c + it * gt
        ht = ot * torch.tanh(ct)
        return torch.cat((ht, ct), dim=1)    # bsz x (2*d_model) x seq_len


class DEQTrellisNet(nn.Module):
    def __init__(self, ninp, nhid, nout, n_layer=40, kernel_size=2, dropouth=0.0,
                 wnorm=True, pretrain_steps=-1, dilation=1):
        super(DEQTrellisNet, self).__init__()

        self.n_layer = n_layer
        self.kernel_size = kernel_size
        self.pretrain_steps = pretrain_steps
        self.func = TrellisNetLSTMLayer(ninp, nhid, nout, kernel_size, dropouth=dropouth, dilation=dilation)
        self.func_copy = copy.deepcopy(self.func)
        self.inject_conv = nn.Conv1d(ninp, self.func.inject_hsize, kernel_size=self.kernel_size)

        if wnorm:
            # If wnorm is specified, we need to make sure to do the deepcopy first
            # because pytorch prevents non-user defined variables from being copied.
            # The deepcopy will only access and copy the `xxx.weight` instead
            # of messing with `xxx.weight_g` that wnorm actually optimizes.
            self.wnorm()
        for params in self.func_copy.parameters():
            params.requires_grad_(False)  # Turn off autograd for func_copy

        self.deq = TrellisNetDEQForward(self.func)
        self.deqback = TrellisNetDEQBackward(self.func, self.func_copy)

    def wnorm(self):
        # Apply weight normalization on both the injection layer and the temporal convolution
        self.inject_conv, self.inject_fn = weight_norm(module=self.inject_conv, names=['weight'], dim=0)
        self.func.wnorm()

    def reset(self, bsz, seq_len):
        self.func.reset(bsz, seq_len)
        self.func_copy.copy(self.func)
        if 'inject_fn' in self.__dict__:
            self.inject_fn.reset(self.inject_conv)

    def transform_input(self, X):
        """
        :param X: Original input sequence of dimension (bsz x embed_dim x seq_len)
        """
        bsz, _, seq_len = X.shape
        total_hsize = self.func.total_hsize

        return torch.zeros(bsz, total_hsize, seq_len).to(X.device)

    def get_output(self, Z):
        """
        Get the output hidden units from intermediate hidden units (different networks should define this differently)

        :param Z: Hidden unit sequence of dimension (bsz x (*)d x seq_len)
        """
        return Z[:, self.func.h_size-self.func.nout:self.func.h_size]

    def get_history(self, Z):
        """
        Get the history repackaging part

        :param Z: Hidden unit sequence of dimension (bsz x (*)d x seq_len)
        """
        return Z[:,:,-1:]

    def forward(self, X, z0, f_thres=50, b_thres=80, train_step=-1, subseq_len=40):
        z1s = self.transform_input(X)  # Dimension (bsz x total_hsize x seq_len)
        bsz, _, seq_len = z1s.shape
        if z0 is None or z0.nelement() == 0:
            z0 = torch.zeros(z1s.size(0), z1s.size(1), 1)
        elif type(z0) == tuple:
            z0 = torch.cat(z0, dim=1)

        # Reset the weight normalization terms and dropout mask; then update the copy at func_copy.
        self.reset(bsz, seq_len)
        
        us = self.inject_conv(F.pad(X, (self.kernel_size-1, 0)))

        if 0 <= train_step < self.pretrain_steps:
            for i in range(self.n_layer):
                z1s = self.func(z1s, us, z0)
        else:
            z1s = self.deq(z1s, us, z0, threshold=f_thres, train_step=train_step, subseq_len=subseq_len)
            if self.training:
                z1s = self.deqback(z1s, us, z0, threshold=b_thres, train_step=train_step, subseq_len=subseq_len)
        out = self.get_output(z1s).transpose(1, 2)   # Dimension (bsz x seq_len x n_out)
        z0 = self.get_history(z1s)                   # Dimension (bsz x total_hsize x 1)

        return out, z0


class DEQTrellisNetLM(nn.Module):
    def __init__(self, n_token, n_layer, ninp, nhid, nout, kernel_size=2,
                 emb_dropout=0.0, dropouti=0.0, dropout=0.0, dropouth=0.0,
                 dropoutl=0.0, wdrop=0.0, wnorm=True, tie_weights=True,
                 pretrain_steps=-1, dilation=1, n_experts=0, load=''):
        super().__init__()
        self.emb_dropout = emb_dropout   # Embedding weight dropout
        self.h_size = nhid + nout
        self.dropout = dropout
        self.dropouti = dropouti
        self.dropoutl = dropoutl         # Mixture-of-Softmax dropout (only valid if MoS is used)
        self.iodrop = VariationalDropout()
        self.trellisnet = DEQTrellisNet(ninp, nhid, nout, n_layer, kernel_size, dropouth=dropouth, wnorm=wnorm,
                                        pretrain_steps=pretrain_steps, dilation=dilation)

        # 1) Set up encoder (embeddings) and decoder (note: an alternative is to use the adaptive embedding; see
        #    DEQ-Transformer implementation)
        self.encoder = nn.Embedding(n_token, ninp)
        self.decoder = nn.Linear(nout, n_token)
        self.init_weights()
        if tie_weights:
            if nout != ninp and self.n_experts == 0:
                raise ValueError('When using the tied flag, nout must be equal to the embedding size')
            self.decoder.weight = self.encoder.weight

        # 2) Set up MoS, if needed
        self.n_experts = n_experts
        if n_experts > 0:
            print("Applied Mixture of Softmax")
            self.mixsoft = MixSoftmax(n_experts, n_token, nlasthid=nout, ninp=ninp, decoder=self.decoder,
                                      dropoutl=dropoutl)
        
        # 3) Load model
        if len(load) > 0:
            params_dict = torch.load(load)
            self.load_weights(params_dict)
            print(f"Finished loading. d_embed={self.trellisnet.inject_conv.weight.data.size(1)}")

        # 4) Apply weight drop connect. If weightnorm is used, we apply the dropout to its "direction", instead of "scale"
        reg_term = '_v' if wnorm else ''
        if wdrop > 0:
            self.trellisnet = WeightDrop(self.trellisnet, [['func', 'full_conv', 'weight' + reg_term],
                                                           ['inject_conv', 'weight' + reg_term]], dropout=wdrop)
            
    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)    
        
    def load_weights(self, params_dict):
        self.load_state_dict(params_dict)
   
    def save_weights(self, name='pretrained_deq'):
        with open(f'{name}.pkl', 'wb') as f:
            print(f"Saving weights at {name}.pkl")
            torch.save(self.state_dict(), f)
            
    def _forward(self, dec_inp, subseq_len, mems=None, f_thres=30, b_thres=40, train_step=-1, decode=True):
        """
        Apply the DEQ-TrellisNet language model on input word tokens

        :param dec_inp: Input words of shape (seq_len x bsz) and dtype torch.LongTensor
        :param subseq_len: The subsequence length with which we feed the segments of the data to DEQ
        :param mems: History padding of dimension (bsz x (*)d x 1); it is passed into the network as z0
        :param f_thres: Forward pass threshold
        :param b_thres: Backward pass threshold
        :param train_step: The number of training step that the current iteration is at
        :return: tuple(output, new memory), where output = tuple(DEQ output, regularized DEQ output, decoded output)
        """
        dec_inp = dec_inp.t()
        bsz = dec_inp.size(0)
        word_emb = embedded_dropout(self.encoder, dec_inp, self.emb_dropout if self.training else 0.0)
        word_emb = self.iodrop(word_emb, self.dropouti).transpose(1, 2)           # (bsz x seq_len x d_model)

        z1s, new_mems = self.trellisnet(word_emb, mems, f_thres=f_thres, b_thres=b_thres, 
                                        train_step=train_step, subseq_len=subseq_len)
        core_out = self.iodrop(z1s, self.dropout)
        decoded = None

        if self.n_experts > 0:
            if not decode: raise ValueError("Mixture of softmax involves decoding phase. Must set decode=True")
            decoded = torch.log(self.mixsoft(core_out).add_(1e-8))

        if decode:
            decoded = decoded if self.n_experts > 0 else self.decoder(core_out)
            return (z1s.transpose(0,1), core_out.transpose(0,1), decoded.transpose(0,1)), [new_mems.permute(2,0,1)]
        
        return (z1s.transpose(0,1), core_out.transpose(0,1), core_out.transpose(0,1)), [new_mems.permute(2,0,1)]

    def forward(self, data, target, mems, train_step=-1, **kwargs):
        bsz = data.size(1)
        if not mems:
            mems = [torch.zeros(1, bsz, 2*self.h_size).to(data.device)]
        mems = mems[0]
        mems = mems.permute(1,2,0)
        subseq_len = kwargs.get('subseq_len', 75)
        decode = kwargs.get('decode', True)
        f_thres = kwargs.get('f_thres', 30)
        b_thres = kwargs.get('b_thres', 40)

        # Note: We can also implement and use the AdaptiveSoftmax for DEQ-TrellisNet here (the same way as in the
        # Transformer network)
        ret = self._forward(data, subseq_len, mems, f_thres=f_thres, b_thres=b_thres, train_step=train_step, decode=decode)
        assert ret[0][0].size(1) == bsz, "Output must have dimension (seq_len x bsz x nout)"
        assert ret[1][0].size(1) == bsz, "New mem must have dimension (1 x bsz x 2*h_size)"
        return ret


if __name__ == '__main__':
    dev = 'cuda'
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    model = DEQTrellisNetLM(n_token=500, n_layer=50, ninp=120, nhid=300, nout=120, kernel_size=2, wnorm=True,
                            tie_weights=True).to(dev)
    raw_data = torch.randint(0, 500, (200, 7)).long().to(dev)  # Generate 500 dummy word tokens (not embeddings)
    data, target = raw_data[:75], raw_data[1:76]
    mems = None
    train_step=-1
    model.eval()

    model.train()
    (_, _, out), mems = model(data, target, mems=mems, f_thres=50, b_thres=80, train_step=train_step)
    loss = out
    loss = loss.float().mean().type_as(loss)
    loss.backward()
    print(model.trellisnet.module.inject_conv.weight_v.grad)   # If not using wnorm, replace weight_v with weight