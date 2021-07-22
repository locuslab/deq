import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.autograd as autograd
import sys
import copy
import numpy as np
from termcolor import colored
import os

sys.path.append('../../')

from lib.optimizations import weight_norm, VariationalDropout, VariationalHidDropout, VariationalAttnDropout
from lib.solvers import anderson, broyden
from lib.jacobian import jac_loss_estimate, power_method

from utils.adaptive_embedding import AdaptiveEmbedding
from utils.positional_embedding import PositionalEmbedding
from utils.proj_adaptive_softmax import ProjectedAdaptiveLogSoftmax
from utils.log_uniform_sampler import LogUniformSampler, sample_logits


class WeightSharePositionwiseFF(nn.Module):
    def __init__(self, d_model, d_inner, dropout, pre_lnorm=False):
        super(WeightSharePositionwiseFF, self).__init__()
        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout

        self.ff1_net = nn.Linear(d_model, d_inner)
        self.drop1 = VariationalHidDropout(dropout=dropout, length_first=True)
        self.ff2_net = nn.Linear(d_inner, d_model)
        self.drop2 = VariationalHidDropout(dropout=dropout, length_first=True)

        self.pre_lnorm = pre_lnorm
    
    def wnorm(self):
        self.ff1_net, self.ff1_fn = weight_norm(module=self.ff1_net, names=['weight'], dim=0)
        self.ff2_net, self.ff2_fn = weight_norm(module=self.ff2_net, names=['weight'], dim=0)

    def reset(self, bsz, qlen):
        self.drop1.reset_mask(bsz, self.d_inner, qlen)
        self.drop2.reset_mask(bsz, self.d_model, qlen)
        if 'ff1_fn' in self.__dict__:
            self.ff1_fn.reset(self.ff1_net)
        if 'ff2_fn' in self.__dict__:
            self.ff2_fn.reset(self.ff2_net)

    def forward(self, inp, attn_out=None):
        assert inp.size(1) == self.d_model, "Feature dimension not match!!"

        inp = inp.transpose(1,2)
        if self.pre_lnorm:
            inp = F.layer_norm(inp, (self.d_model,))
        relu_out1 = self.drop1(F.relu(self.ff1_net(inp)))
        out2 = self.drop2(self.ff2_net(relu_out1))
        output = out2 + inp
        if not self.pre_lnorm:
            output = F.layer_norm(output, (self.d_model,))
        return output.transpose(1,2)


class WeightShareSelfAttention(nn.Module):
    # This is similar to the RelPartialLearnableMultiHeadAttn class in Transformer-XL
    def __init__(self, d_model, n_head, d_head, dropout, dropatt, 
                 pre_lnorm=False, local_size=None):
        super(WeightShareSelfAttention, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head
        self.dropout = dropout
        self.scale = 1 / (d_head ** 0.5)

        self.qkv_net = nn.Conv1d(d_model, 3 * n_head * d_head, kernel_size=1, bias=False)
        self.r_net = nn.Conv1d(d_model, n_head * d_head, kernel_size=1, bias=False)
        self.r_w_bias = nn.Parameter(torch.rand(n_head, d_head).uniform_(-0.05, 0.05))
        self.r_r_bias = nn.Parameter(torch.rand(n_head, d_head).uniform_(-0.05, 0.05))
        self.o_net = nn.Conv1d(n_head * d_head, d_model, kernel_size=1)
        self.dropatt = VariationalAttnDropout(dropout=dropatt)
        self.drop = VariationalHidDropout(dropout=dropout)

        self.pre_lnorm = pre_lnorm
        self.local_size = local_size
        
    def wnorm(self):
        self.qkv_net, self.qkv_fn = weight_norm(module=self.qkv_net, names=['weight'], dim=0)
        self.r_net, self.r_fn = weight_norm(module=self.r_net, names=['weight'], dim=0)
        self.o_net, self.o_fn = weight_norm(module=self.o_net, names=['weight'], dim=0)

    def reset(self, bsz, qlen, klen):
        self.dropatt.reset_mask(bsz, self.n_head, qlen, klen)
        self.drop.reset_mask(bsz, self.d_model, qlen)
        if 'qkv_fn' in self.__dict__:
            self.qkv_fn.reset(self.qkv_net)
        if 'r_fn' in self.__dict__:
            self.r_fn.reset(self.r_net)
        if 'o_fn' in self.__dict__:
            self.o_fn.reset(self.o_net)

    def _rel_shift(self, x):
        # x has dimension (bsz x n_head x qlen x klen)
        bsz, n_head, qlen, klen = x.size()
        x_padded = F.pad(x, (1,0))
        x_padded = x_padded.view(bsz, n_head, klen+1, qlen)
        return x_padded[:,:,1:].view_as(x)

    def forward(self, z1ss, pos_emb, u1ss, mems=None):
        # Note: In this context, qlen means the length of the sequence; and mlen describes
        #       the length of the padding. Their sum is klen. 
        bsz, d_model, qlen = z1ss.size()
        r_w_bias, r_r_bias = self.r_w_bias, self.r_r_bias
        n_head, d_head = self.n_head, self.d_head
        rlen = pos_emb.size(2)
        
        if mems is None: 
            mems = torch.tensor([]).view(0,0,0)
        mlen = mems.size(2)
        cat = torch.cat([mems, z1ss], dim=-1)

        if self.pre_lnorm:
            cat = F.layer_norm(cat.transpose(1,2), (d_model,)).transpose(1,2)
        w_heads = self.qkv_net(cat)      # (N x 3*d_model x seq_len)
        r_head_k = self.r_net(pos_emb)

        # Input injection
        w_heads += u1ss
        w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=1)
        w_head_q = w_head_q[:,:,-qlen:]

        klen = w_head_k.size(2)

        w_head_q = w_head_q.view(bsz, n_head, d_head, qlen)           # bsz x n_head x d_head x qlen
        w_head_k = w_head_k.view(bsz, n_head, d_head, klen)           # bsz x n_head x d_head x klen
        w_head_v = w_head_v.view(bsz, n_head, d_head, klen)           # bsz x n_head x d_head x klen

        r_head_k = r_head_k.view(n_head, d_head, rlen)                # n_head x d_head x rlen

        # Compute attention score
        rw_head_q = w_head_q + r_w_bias[:,:,None]                   # bsz x n_head x d_head x qlen
        AC = torch.einsum('bndi,bndj->bnij', rw_head_q, w_head_k)
        rr_head_q = w_head_q + r_r_bias[:,:,None]
        BD = torch.einsum('bndi,ndj->bnij', rr_head_q, r_head_k)
        BD = self._rel_shift(BD)    # for relative positional embedding

        attn_score = AC + BD        # bsz x n_head x qlen x klen
        attn_score.mul_(self.scale)
            
        # Compute attention probability
        # We apply a local mask, with local horizon size of mlen
        local_size = self.local_size or 1000
        attn_mask = (torch.triu(torch.ones(qlen, klen), diagonal=1+mlen) > 0)[None]
        attn_mask += (torch.tril(torch.ones(qlen, klen), diagonal=mlen-local_size) > 0)[None]
        if attn_mask is not None and attn_mask.any().item():
            attn_score = attn_score.float().masked_fill(
                    attn_mask[None], -float('inf')).type_as(attn_score)
                
        attn_prob = F.softmax(attn_score, dim=-1)          # bsz x n_head x qlen x klen
        attn_prob = self.dropatt(attn_prob)
            
        # Compute attention vector
        attn_vec = torch.einsum('bnij,bndj->bndi', (attn_prob, w_head_v))
        
        # [bsz x d x qlen]
        attn_vec = attn_vec.contiguous().view(bsz, n_head*d_head, attn_vec.size(-1))

        # Linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)
        
        # Residual connection + layer normolization (if applicable)
        if self.pre_lnorm:
            out = attn_out + z1ss
        else:
            out = F.layer_norm((attn_out + z1ss).transpose(1,2), (d_model,)).transpose(1,2)
        return out


class RelPartialLearnableDecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout, **kwargs):
        super(RelPartialLearnableDecoderLayer, self).__init__()

        pre_lnorm = kwargs.get('pre_lnorm')
        local_size = kwargs.get('local_size', None)
        dropatt = kwargs.get('dropatt', 0.0)
        self.dec_attn = WeightShareSelfAttention(d_model, n_head, d_head, dropout=dropout, dropatt=dropatt, 
                                                 pre_lnorm=pre_lnorm,  local_size=local_size)
        self.pos_ff = WeightSharePositionwiseFF(d_model, d_inner, dropout, pre_lnorm=pre_lnorm)
    
    def wnorm(self):
        self.dec_attn.wnorm()
        self.pos_ff.wnorm()

    def reset(self, bsz, qlen, klen):
        # Reset the dropout mask(s) and re-compute the weight normalized weights at the START of each iterations
        self.dec_attn.reset(bsz, qlen, klen)
        self.pos_ff.reset(bsz, qlen)

    def forward(self, z1ss, uss, z0, *args):
        pos_emb = args[0]
        output = self.dec_attn(z1ss, pos_emb, uss, mems=z0)
        output = self.pos_ff(output)
        return output


class DEQTransformerLM(nn.Module):
    def __init__(self, n_token, n_layer, eval_n_layer, n_head, d_model, d_head, d_inner,
                 dropout, dropatt, tie_weights=True, d_embed=None, div_val=1,
                 tie_projs=[False], pre_lnorm=False, wnorm=False, tgt_len=None,
                 mem_len=None, local_size=0, pretrain_steps=1, cutoffs=[], load='', 
                 f_solver=anderson, b_solver=None, stop_mode="rel", logging=None):
        super().__init__()
        self.n_token = n_token

        d_embed = d_model if d_embed is None else d_embed
        self.d_embed = d_embed
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head

        self.word_emb = AdaptiveEmbedding(n_token, d_embed, d_model, cutoffs, div_val=div_val)
        self.iodrop = VariationalDropout()
        self.dropout = dropout
        self.pos_drop = VariationalHidDropout(dropout=dropout)
        self.pretrain_steps = pretrain_steps

        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.local_size = local_size
        self.max_klen = tgt_len + mem_len

        self.n_layer = n_layer
        self.eval_n_layer = eval_n_layer
        self.inject_conv = nn.Conv1d(d_model, 3*d_model, kernel_size=1)
        self.pos_emb = PositionalEmbedding(self.d_model)
        self.func = RelPartialLearnableDecoderLayer(n_head, d_model, d_head, d_inner, dropout=dropout, dropatt=dropatt,
                                                    pre_lnorm=pre_lnorm, local_size=local_size)
        self.f_solver = f_solver
        self.b_solver = b_solver if b_solver else self.f_solver
        self.hook = None
        self.stop_mode = stop_mode
        self.alternative_mode = "abs" if self.stop_mode == "rel" else "rel"
        self.logging = logging or print
        if wnorm: self.func.wnorm()

        # use adaptive softmax (including standard softmax)
        # (Note: To use sample softmax, refer to the Transformer-XL implementation)
        self.crit = ProjectedAdaptiveLogSoftmax(n_token, d_embed, d_model, cutoffs, div_val=div_val)

        if tie_weights:
            for i in range(len(self.crit.out_layers)):
                self.crit.out_layers[i].weight = self.word_emb.emb_layers[i].weight

        if tie_projs:
            for i, tie_proj in enumerate(tie_projs):
                if tie_proj and div_val == 1 and d_model != d_embed:
                    self.crit.out_projs[i].weight.data = self.word_emb.emb_projs[0].weight.data
                elif tie_proj and div_val != 1:
                    self.crit.out_projs[i].weight.data = self.word_emb.emb_projs[i].weight.data

        if len(load) > 0:
            params_dict = torch.load(load)
            self.load_weights(params_dict)
            self.logging(f"Finished loading. d_embed={self.inject_conv.weight.data.size(1)}")

    def reset_length(self, tgt_len, mem_len):
        self.tgt_len = tgt_len
        self.mem_len = mem_len

    def load_weights(self, params_dict):
        self.load_state_dict(params_dict)
   
    def save_weights(self, path, name='pretrained_deq'):
        with open(os.path.join(path, f'{name}.pth'), 'wb') as f:
            self.logging(f"Saving weight state dict at {name}.pth")
            torch.save(self.state_dict(), f)

    def init_mems(self):
        if self.mem_len <= 0:
            self.logging("init_mems: Hmmmm... you shouldn't be here.")
            return None

        # mems is not None
        with torch.no_grad():
            mems = [torch.empty(0), torch.empty(0)]
            return mems       # For z0 and u0
            
    def _update_mems(self, z1s, us, z0, qlen, mlen):
        # does not deal with None
        if self.mem_len <= 0: 
            self.logging("_update_mems: Hmmmm... you shouldn't be here.")
            return None

        # mems is not None
        with torch.no_grad():
            end_idx = mlen + qlen
            beg_idx = max(0, end_idx - self.mem_len)   # Account for when mlen = 0
            zs = torch.cat([z0, z1s], dim=2)
            new_z0 = zs[:,:,beg_idx:end_idx].detach().permute(2,0,1).contiguous()     # seq_len x bsz x d_model
            new_u0 = us[:,:,beg_idx:end_idx].detach().permute(2,0,1).contiguous()

            return [new_z0, new_u0]

    def _forward(self, dec_inp, mems=None, f_thres=30, b_thres=40, train_step=-1,
                 compute_jac_loss=True, spectral_radius_mode=False, writer=None):
        """
        Apply the DEQ-Transformer language model on input word tokens

        :param dec_inp: Input words of shape (seq_len x bsz) and dtype torch.LongTensor
        :param mems: History madding and the transformed input corresponding to it; must be a tuple (z0, u0)
                     where z0 has dimension (bsz x d_model x pad_len) and u0 has size (bsz x 3*d_model x pad_len)
        :param f_thres: Forward pass threshold
        :param b_thres: Backward pass threshold
        :param train_step: The number of training step that the current iteration is at
        :param compute_jac_loss: Whether to return an (optional) Jacobian-stability-related loss
        :param spectral_radius_mode: Whether to estimate spectral radius at J(z*) (note: this is very slow!!)
        :param writer: Tensorboard writer
        :return: tuple(output sequence, new memory, jac loss, spec. radius)
        """
        # Assume dec_inp has shape (qlen x bsz)
        dec_inp = dec_inp.t()                              
        bsz, qlen = dec_inp.size()
        word_emb = self.word_emb(dec_inp)
        word_emb = self.iodrop(word_emb, self.dropout)
        u1s = self.inject_conv(word_emb.transpose(1,2))      # bsz x 3*d_model x qlen

        z0, u0 = mems
        d_model = self.d_model
        if z0 is not None and z0.nelement() > 0:
            assert z0.size(2) == u0.size(2), "Padding fixed points and padding embedding dimensions don't agree"
        else:
            z0, u0 = torch.zeros(bsz, d_model, 0), torch.zeros(bsz, 3*d_model, 0)
        mlen = z0.size(2)
        klen = mlen + qlen    # qlen is seq_len, mlen is pad_len

        pos_seq = torch.arange(klen-1, -1, -1.0)
        pos_emb = self.pos_drop(self.pos_emb(pos_seq))     # bsz x d_model x (qlen + mlen) for positional embedding
        us = torch.cat([u0, u1s], dim=2)
        z1s = torch.zeros(bsz, d_model, qlen)          # bsz x d_model x (qlen + mlen) for initial estimate of output
        func_args = [us, z0, pos_emb]
        jac_loss = torch.tensor(0.0).to(z1s)
        sradius = torch.zeros(bsz, 1).to(z1s)
        deq_mode = (train_step < 0) or (train_step >= self.pretrain_steps)

        if not deq_mode:
            # In pretraining mode with stacked (weight-tied) layers. NOT recommended for large models (as then
            # a stacking of, for example, 16 layers would be extremely inefficient). One can also train with
            # M layers and evaluate using N layers (which typically leads to worse performance).
            n_layer = self.n_layer if self.training or train_step > 0 else self.eval_n_layer
            for i in range(n_layer):
                z1s = self.func(z1s, *func_args)
            new_z1s = z1s
        else:
            # Compute the equilibrium via DEQ. When in training mode, we need to register the analytical backward
            # pass according to the Theorem 1 in the paper.
            with torch.no_grad():
                result = self.f_solver(lambda z: self.func(z, *func_args), z1s, threshold=f_thres, stop_mode=self.stop_mode)
                z1s = result['result']
            new_z1s = z1s

            if (not self.training) and spectral_radius_mode:
                with torch.enable_grad():
                    z1s.requires_grad_()
                    new_z1s = self.func(z1s, *func_args)
                _, sradius = power_method(new_z1s, z1s, n_iters=150)
            
            if self.training:
                z1s.requires_grad_()
                new_z1s = self.func(z1s, *func_args)
                if compute_jac_loss:
                    jac_loss = jac_loss_estimate(new_z1s, z1s, vecs=1)

                def backward_hook(grad):
                    if self.hook is not None:
                        # To avoid infinite loop
                        self.hook.remove()
                        torch.cuda.synchronize()
                    new_grad = self.b_solver(lambda y: autograd.grad(new_z1s, z1s, y, retain_graph=True)[0] + grad, \
                                             torch.zeros_like(grad), threshold=b_thres)['result']
                    return new_grad
                self.hook = new_z1s.register_hook(backward_hook)

        core_out = self.iodrop(new_z1s, self.dropout).permute(2,0,1).contiguous()       # qlen x bsz x d_model
        new_mems = self._update_mems(new_z1s, us, z0, mlen, qlen)
        return core_out, new_mems, jac_loss.view(-1,1), sradius.view(-1,1)

    def forward(self, data, target, mems, train_step=-1, **kwargs):
        # nn.DataParallel does not allow size(0) tensors to be broadcasted.
        # So, have to initialize size(0) mems inside the model forward.
        # Moreover, have to return new_mems to allow nn.DataParallel to piece
        # them together.
        if not mems: 
            mems = self.init_mems()
        else:
            for i in range(len(mems)):
                mems[i] = mems[i].permute(1,2,0).contiguous()        # bsz x [-1] x seq_len
        qlen, bsz = data.size()
        mlen = 0 if mems[0].nelement() == 0 else mems[0].size(2)
        klen = mlen + qlen
    
        # Reset dropout in self.func
        self.pos_drop.reset_mask(1, self.d_model, klen)
        self.func.reset(bsz, qlen, klen)

        tgt_len = target.size(0)
        f_thres = kwargs.get('f_thres', 30)
        b_thres = kwargs.get('b_thres', 40)
        compute_jac_loss = kwargs.get('compute_jac_loss', True)
        sradius_mode = kwargs.get('spectral_radius_mode', False)
        writer = kwargs.get('writer', None)
        hidden, new_mems, jac_loss, sradius = self._forward(data, mems=mems, f_thres=f_thres, b_thres=b_thres, train_step=train_step, 
                                                            compute_jac_loss=compute_jac_loss, spectral_radius_mode=sradius_mode, 
                                                            writer=writer)
        pred_hid = hidden[-tgt_len:]
        loss = self.crit(pred_hid.view(-1, pred_hid.size(-1)), target.contiguous().view(-1))
        loss = loss.view(tgt_len, -1)

        if new_mems is None:
            return [loss, jac_loss, sradius]
        else:
            return [loss, jac_loss, sradius] + new_mems
