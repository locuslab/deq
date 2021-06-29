import torch
from torch.autograd import Function


class Identity(Function):
    @staticmethod
    def forward(ctx, x, name):
        ctx.name = name
        return x.clone()

    def backward(ctx, grad):
        import pydevd
        pydevd.settrace(suspend=False, trace_only_current_thread=True)
        grad_temp = grad.clone()
        return grad_temp, None