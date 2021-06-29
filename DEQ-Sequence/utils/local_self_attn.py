import torch
import lsa_cuda
import localdot_cuda
import pickle

class LSA(torch.autograd.Function):

    @staticmethod
    def forward(ctx, qs, ks, horizon=10):
        ctx.save_for_backward(qs, ks)
        ctx.horizon = horizon
        return lsa_cuda.forward(qs, ks, horizon)

    @staticmethod
    def backward(ctx, grad_qk):
        qs, ks = ctx.saved_tensors
        horizon = ctx.horizon
        grad_qk = grad_qk.clone()
        res = lsa_cuda.backward(grad_qk, qs, ks, horizon)
        return res[0], res[1], None
    

class Localdot(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, qks, vs):
        ctx.save_for_backward(qks, vs)
        return localdot_cuda.forward(qks, vs)
   
    @staticmethod
    def backward(ctx, grad_qkv):
        qks, vs = ctx.saved_tensors
        grad_qkv = grad_qkv.clone()
        res = localdot_cuda.backward(grad_qkv, qks, vs)
        # print(res[1][0,0,-10:])
        # print(res[1][0,0].shape)
        return res[0], res[1]
    
    
class Identity(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x+1-1
   
    @staticmethod
    def backward(ctx, grad_y):
        x = ctx.saved_tensors
        res = grad_y
        # print(res[1,1])
        # print(res.permute(2,3,0,1)[1,1])
        print(res.permute(1,2,0,3).mean(0).mean(0))
        # print(res[:,4])
        return res