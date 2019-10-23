import torch
import torch.nn.functional as F
import torch.nn as nn
import sys
import copy

sys.path.append('../../')

from modules.optimizations import weight_norm, VariationalDropout, VariationalHidDropout, VariationalAttnDropout

from models.transformers.deq_transformer_forward_backward import TransformerDEQForward, TransformerDEQBackward

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

        self.ff1_net = nn.Conv1d(d_model, d_inner, kernel_size=1)
        self.drop1 = VariationalHidDropout(dropout=dropout)
        self.ff2_net = nn.Conv1d(d_inner, d_model, kernel_size=1)
        self.drop2 = VariationalHidDropout(dropout=dropout)

        self.pre_lnorm = pre_lnorm
    
    def wnorm(self):
        print("Weight normalization applied to PosFF")
        self.ff1_net, self.ff1_fn = weight_norm(module=self.ff1_net, names=['weight'], dim=0)
        self.ff2_net, self.ff2_fn = weight_norm(module=self.ff2_net, names=['weight'], dim=0)

    def reset(self, bsz, qlen):
        self.drop1.reset_mask(torch.zeros(bsz, self.d_inner, qlen))
        self.drop2.reset_mask(torch.zeros(bsz, self.d_model, qlen))
        if 'ff1_fn' in self.__dict__:
            self.ff1_fn.reset(self.ff1_net)
        if 'ff2_fn' in self.__dict__:
            self.ff2_fn.reset(self.ff2_net)

    def copy(self, func):
        self.ff1_net.weight.data = func.ff1_net.weight.data.clone()
        self.ff2_net.weight.data = func.ff2_net.weight.data.clone()
        self.ff1_net.bias.data = func.ff1_net.bias.data.clone()
        self.ff2_net.bias.data = func.ff2_net.bias.data.clone()
        self.drop1.mask = func.drop1.mask.clone()
        self.drop2.mask = func.drop2.mask.clone()

    def forward(self, inp, attn_out=None):
        assert inp.size(1) == self.d_model, "Feature dimension not match!!"

        if self.pre_lnorm:
            inp = F.layer_norm(inp.transpose(1,2), (self.d_model,)).transpose(1,2)
        relu_out1 = self.drop1(F.relu(self.ff1_net(inp)))
        out2 = self.drop2(self.ff2_net(relu_out1))
        output = out2 + inp
        if not self.pre_lnorm:
            output = F.layer_norm(output.transpose(1,2), (self.d_model,)).transpose(1,2)
        return output


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
        print("Weight normalization applied to SA")
        self.qkv_net, self.qkv_fn = weight_norm(module=self.qkv_net, names=['weight'], dim=0)
        self.r_net, self.r_fn = weight_norm(module=self.r_net, names=['weight'], dim=0)
        self.o_net, self.o_fn = weight_norm(module=self.o_net, names=['weight'], dim=0)

    def reset(self, bsz, qlen, klen):
        self.dropatt.reset_mask(torch.zeros(bsz, self.n_head, qlen, klen))
        self.drop.reset_mask(torch.zeros(bsz, self.d_model, qlen))
        if 'qkv_fn' in self.__dict__:
            self.qkv_fn.reset(self.qkv_net)
        if 'r_fn' in self.__dict__:
            self.r_fn.reset(self.r_net)
        if 'o_fn' in self.__dict__:
            self.o_fn.reset(self.o_net)

    def copy(self, func):
        # Destructive copy
        self.qkv_net.weight.data = func.qkv_net.weight.data.clone()
        self.r_net.weight.data = func.r_net.weight.data.clone()
        self.r_w_bias.data = func.r_w_bias.data.clone()
        self.r_r_bias.data = func.r_r_bias.data.clone()
        self.o_net.weight.data = func.o_net.weight.data.clone()
        self.o_net.bias.data = func.o_net.bias.data.clone()
        self.dropatt.mask = func.dropatt.mask.clone()
        self.drop.mask = func.drop.mask.clone()

    def _rel_shift(self, x):
        # x has dimension (bsz x n_head x qlen x klen)
        bsz, n_head, qlen, klen = x.size()
        x_padded = F.pad(x, (1,0))
        x_padded = x_padded.view(bsz, n_head, klen+1, qlen)
        return x_padded[:,:,1:].view_as(x)

    def forward(self, z1ss, pos_emb, u1ss, mems=None):
        # Note: In this context, qlen means the length of the (small) subsequence; and mlen describes
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
        self.dec_attn = WeightShareSelfAttention(d_model, n_head, d_head, dropout,
                                                 dropatt=0.0, pre_lnorm=pre_lnorm,  local_size=local_size)
        self.pos_ff = WeightSharePositionwiseFF(d_model, d_inner, dropout, pre_lnorm=pre_lnorm)
    
    def wnorm(self):
        self.dec_attn.wnorm()
        self.pos_ff.wnorm()

    def reset(self, bsz, qlen, klen):
        # Reset the dropout mask(s) and re-compute the weight normalized weights at the START of each iterations
        self.dec_attn.reset(bsz, qlen, klen)
        self.pos_ff.reset(bsz, qlen)

    def copy(self, func):
        self.dec_attn.copy(func.dec_attn)
        self.pos_ff.copy(func.pos_ff)

    def forward(self, z1ss, uss, z0, *args):
        pos_emb = args[0]
        output = self.dec_attn(z1ss, pos_emb, uss, mems=z0)
        output = self.pos_ff(output)
        return output


class DEQTransformerLM(nn.Module):
    def __init__(self, n_token, n_layer, eval_n_layer, n_head, d_model, d_head, d_inner,
                 dropout, dropatt, tie_weights=True, d_embed=None, div_val=1,
                 tie_projs=[False], pre_lnorm=False, wnorm=False, tgt_len=None,
                 mem_len=None, local_size=0, pretrain_steps=1, cutoffs=[], load=''):
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
        self.drop = VariationalHidDropout(dropout=dropout)
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
        self.func_copy = copy.deepcopy(self.func)
        if wnorm:
            # If wnorm is specified, we need to make sure to do the deepcopy first
            # because pytorch prevents non-user defined variables from being copied.
            # The deepcopy will only access and copy the `xxx.weight` instead
            # of messing with `xxx.weight_g` that wnorm actually optimizes.
            self.func.wnorm()
        for params in self.func_copy.parameters():
            params.requires_grad_(False)                 # Turn off autograd for func_copy
        self.deq = TransformerDEQForward(self.func)
        self.deqback = TransformerDEQBackward(self.func, self.func_copy)

        # use adaptive softmax (including standard softmax)
        # (Note: To use sample softmax, refer to the Transformer-XL implementation)
        self.crit = ProjectedAdaptiveLogSoftmax(n_token, d_embed, d_model, cutoffs, div_val=div_val)

        if tie_weights:
            for i in range(len(self.crit.out_layers)):
                self.crit.out_layers[i].weight = self.word_emb.emb_layers[i].weight

        if tie_projs:
            for i, tie_proj in enumerate(tie_projs):
                if tie_proj and div_val == 1 and d_model != d_embed:
                    self.crit.out_projs[i] = self.word_emb.emb_projs[0]
                elif tie_proj and div_val != 1:
                    self.crit.out_projs[i] = self.word_emb.emb_projs[i]

        if len(load) > 0:
            params_dict = torch.load(load)
            self.load_weights(params_dict)
            print(f"Finished loading. d_embed={self.inject_conv.weight.data.size(1)}")

    def reset_length(self, tgt_len, mem_len):
        self.tgt_len = tgt_len
        self.mem_len = mem_len

    def load_weights(self, params_dict):
        self.load_state_dict(params_dict)
   
    def save_weights(self, name='pretrained_deq'):
        with open(f'{name}.pkl', 'wb') as f:
            print(f"Saving weights at {name}.pkl")
            torch.save(self.state_dict(), f)

    def init_mems(self):
        if self.mem_len <= 0:
            print("init_mems: Hmmmm... you shouldn't be here.")
            return None

        # mems is not None
        with torch.no_grad():
            param = next(self.parameters()) 
            mems = [torch.empty(0, dtype=param.dtype, device=param.device),
                    torch.empty(0, dtype=param.dtype, device=param.device)]
            return mems       # For z0 and u0
            
    def _update_mems(self, z1s, us, z0, qlen, mlen):
        # does not deal with None
        if self.mem_len <= 0: 
            print("_update_mems: Hmmmm... you shouldn't be here.")
            return None

        # mems is not None
        with torch.no_grad():
            end_idx = mlen + qlen
            beg_idx = max(0, end_idx - self.mem_len)   # Account for when mlen = 0
            zs = torch.cat([z0, z1s], dim=2)
            new_z0 = zs[:,:,beg_idx:end_idx].detach().permute(2,0,1).contiguous()     # seq_len x bsz x d_model
            new_u0 = us[:,:,beg_idx:end_idx].detach().permute(2,0,1).contiguous()

            return [new_z0, new_u0]

    def _forward(self, dec_inp, subseq_len, mems=None, f_thres=30, b_thres=40, train_step=-1):
        """
        Apply the DEQ-Transformer language model on input word tokens

        :param dec_inp: Input words of shape (seq_len x bsz) and dtype torch.LongTensor
        :param subseq_len: The subsequence length with which we feed the segments of the data to DEQ
        :param mems: History madding and the transformed input corresponding to it; must be a tuple (z0, u0)
                     where z0 has dimension (bsz x d_model x pad_len) and u0 has size (bsz x 3*d_model x pad_len)
        :param f_thres: Forward pass threshold
        :param b_thres: Backward pass threshold
        :param train_step: The number of training step that the current iteration is at
        :return: tuple(output sequence, new memory)
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
            z0 = torch.zeros(bsz, d_model, 0)
            u0 = torch.zeros(bsz, 3*d_model, 0)
        mlen = z0.size(2)
        klen = mlen + qlen    # qlen is seq_len, mlen is pad_len

        pos_seq = torch.arange(klen-1, -1, -1.0)
        pos_emb = self.drop(self.pos_emb(pos_seq))     # bsz x d_model x (qlen + mlen) for positional embedding
        us = torch.cat([u0, u1s], dim=2)
        z1s = torch.zeros(bsz, d_model, qlen)          # bsz x d_model x (qlen + mlen) for initial estimate of output

        if 0 <= train_step < self.pretrain_steps:
            # In pretraining mode with stacked (weight-tied) layers. NOT recommended for large models (as then
            # a stacking of, for example, 16 layers would be extremely inefficient). One can also train with
            # M layers and evaluate using N layers (which typically leads to worse performance).
            n_layer = self.n_layer if self.training or train_step > 0 else self.eval_n_layer
            torch.cuda.empty_cache()
            for i in range(n_layer):
                z1s = self.func(z1s, us, z0, pos_emb)
        else:
            # Compute the equilibrium via DEQ. When in training mode, we need to register the analytical backward
            # pass according to the Theorem 1 in the paper.
            z1s = self.deq(z1s, us, z0, pos_emb=pos_emb, subseq_len=subseq_len, threshold=f_thres, train_step=train_step)
            if self.training:
                z1s = self.deqback(z1s, us, z0, pos_emb=pos_emb, subseq_len=subseq_len, threshold=b_thres, train_step=train_step)
                    
        core_out = self.iodrop(z1s, self.dropout)
        core_out = core_out.permute(2,0,1).contiguous()       # qlen x bsz x d_model
        new_mems = self._update_mems(z1s, us, z0, mlen, qlen)
        return core_out, new_mems

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
        
        # Reset dropout in self.func, and copy everything (weights, dropout masks) to self.func_copy
        self.drop.reset_mask(torch.zeros(1, self.d_model, klen))
        self.func.reset(bsz, qlen, klen)
        self.func_copy.copy(self.func)

        tgt_len = target.size(0)
        subseq_len = kwargs.get('subseq_len', 75)
        f_thres = kwargs.get('f_thres', 30)
        b_thres = kwargs.get('b_thres', 40)
        hidden, new_mems = self._forward(data, subseq_len=subseq_len, mems=mems, 
                                         f_thres=f_thres, b_thres=b_thres, train_step=train_step)
        
        pred_hid = hidden[-tgt_len:]
        loss = self.crit(pred_hid.view(-1, pred_hid.size(-1)), target.view(-1))
        loss = loss.view(tgt_len, -1)

        if new_mems is None:
            return [loss]
        else:
            return [loss] + new_mems


if __name__ == '__main__':
    dev = 'cuda'
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    model = DEQTransformerLM(n_token=500, n_layer=2,
                             eval_n_layer=24, n_head=12, d_model=120, d_head=10, d_inner=500,
                             dropout=0.1, dropatt=0.1, mem_len=100, tgt_len=100, tie_weights=True, d_embed=None).to(dev)
    raw_data = torch.randint(0, 500, (200, 7)).long().to(dev)
    data, target = raw_data[:75], raw_data[1:76]
    mems = None
    train_step=-1
    model.eval()

    model.train()
    ret = model(data, target, mems=mems, f_thres=50, b_thres=80, train_step=train_step)
    loss, mems = ret[0], ret[1:]
    loss = loss.float().mean().type_as(loss)
    loss.backward()
    print(model.func.dec_attn.qkv_net.weight.grad)