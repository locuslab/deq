import torch
from torch import nn
import torch.nn.functional as functional
from torch.autograd import Function
import numpy as np
import pickle
import sys
import os
from scipy.optimize import root
import time
import copy
from modules.broyden import broyden, analyze_broyden


class RootFind(Function):
    """ Generic DEQ module that uses Broyden's method to find the equilibrium state """
    @staticmethod
    def f(func, z1ss, uss, z0, *args):
        return func(z1ss, uss, z0, *args)

    @staticmethod
    def g(func, z1ss, uss, z0, *args):
        return RootFind.f(func, z1ss, uss, z0, *args) - z1ss

    @staticmethod
    def broyden_find_root(func, z1ss, uss, z0, eps, *args):
        bsz, d_model, seq_len = z1ss.size()
        z1ss_est = z1ss.clone().detach()
        threshold = args[-2]    # Can also set this to be different, based on training/inference
        train_step = args[-1]

        g = lambda x: RootFind.g(func, x, uss, z0, *args)
        result_info = broyden(g, z1ss_est, threshold=threshold, eps=eps, name="forward")
        z1ss_est = result_info['result']
            
        if threshold > 100:
            torch.cuda.empty_cache()
        return z1ss_est.clone().detach()

    @staticmethod
    def forward(ctx, func, z1ss, uss, z0, *args):
        bsz, d_model, seq_len = z1ss.size()
        eps = 1e-6 * np.sqrt(bsz * seq_len * d_model)
        root_find = RootFind.broyden_find_root
        ctx.args_len = len(args)
        with torch.no_grad():
            z1ss_est = root_find(func, z1ss, uss, z0, eps, *args)   # args include pos_emb, threshold, train_step

            # If one would like to analyze the convergence process (e.g., failures, stability), should
            # insert here or in broyden_find_root.
            return z1ss_est

    @staticmethod
    def backward(ctx, grad_z1):
        grad_args = [None for _ in range(ctx.args_len)]
        return (None, grad_z1, None, None, *grad_args)

    
class DEQModule(nn.Module):

    """ 
    The equilibrium solver module. Forward pass is unspecified; we provide an implementation of the
    implicit differentiation through the equilibrium point in the inner `Backward` class.
    """

    def __init__(self, func, func_copy):
        super(DEQModule, self).__init__()
        self.func = func
        self.func_copy = func_copy

    def forward(self, z1s, us, z0, **kwargs):
        raise NotImplemented

    class Backward(Function):
        """
        A 'dummy' function that does nothing in the forward pass and perform implicit differentiation
        in the backward pass. Essentially a wrapper that provides backprop for the `DEQModule` class.
        You should use this inner class in DEQModule's forward() function by calling:
        
            self.Backward.apply(self.func_copy, ...)
            
        """
        @staticmethod
        def forward(ctx, func_copy, z1ss, uss, z0, *args):
            ctx.save_for_backward(z1ss, uss, z0)
            ctx.func = func_copy
            ctx.args = args
            return z1ss

        @staticmethod
        def backward(ctx, grad):
            torch.cuda.empty_cache()

            # grad should have dimension (bsz x d_model x seq_len)
            bsz, d_model, seq_len = grad.size()
            grad = grad.clone()
            z1ss, uss, z0 = ctx.saved_tensors
            args = ctx.args
            threshold, train_step = args[-2:]

            func = ctx.func
            z1ss = z1ss.clone().detach().requires_grad_()
            uss = uss.clone().detach()
            z0 = z0.clone().detach()

            with torch.enable_grad():
                y = RootFind.g(func, z1ss, uss, z0, *args)

            def g(x):
                y.backward(x, retain_graph=True)   # Retain for future calls to g
                JTx = z1ss.grad.clone().detach()
                z1ss.grad.zero_()
                return JTx + grad

            eps = 2e-10 * np.sqrt(bsz * seq_len * d_model)
            dl_df_est = torch.zeros_like(grad)

            result_info = broyden(g, dl_df_est, threshold=threshold, eps=eps, name="backward")
            dl_df_est = result_info['result']
            
            y.backward(torch.zeros_like(dl_df_est), retain_graph=False)

            grad_args = [None for _ in range(len(args))]
            return (None, dl_df_est, None, None, *grad_args)
