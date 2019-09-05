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
from termcolor import colored
import copy
from modules.broyden import broyden, analyze_broyden


class DEQFunc(Function):
    """ Generic DEQ module that uses Broyden's method to find the equilibrium state """
    @staticmethod
    def f(func, z1ss, uss, z0, *args):
        return func(z1ss, uss, z0, *args)

    @staticmethod
    def g(func, z1ss, uss, z0, *args):
        return DEQFunc.f(func, z1ss, uss, z0, *args) - z1ss

    @staticmethod
    def broyden_find_root(func, z1ss, uss, z0, eps, *args):
        bsz, d_model, seq_len = z1ss.size()
        z1ss_est = z1ss.clone().detach()
        threshold = args[-2]    # Can also set this to be different, based on training/inference
        train_step = args[-1]

        g = lambda x: DEQFunc.g(func, x, uss, z0, *args)
        result_info = broyden(g, z1ss_est, threshold=threshold, eps=eps, name="forward")
        z1ss_est = result_info['result']
        nstep = result_info['nstep']

        if threshold > 100:
            torch.cuda.empty_cache()
        return z1ss_est.clone().detach()

    @staticmethod
    def forward(ctx, func, z1ss, uss, z0, *args):
        bsz, d_model, seq_len = z1ss.size()
        eps = 1e-6 * np.sqrt(bsz * seq_len * d_model)
        root_find = DEQFunc.broyden_find_root
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


class DummyDEQFunc(Function):
    """ This module is created only to make backward implementation easier. """
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
        threshold = args[-2]
        train_step = args[-1]

        func = ctx.func
        z1ss_temp = z1ss.clone().detach().requires_grad_()
        uss_temp = uss.clone().detach()
        z0_temp = z0.clone().detach()
        args_temp = copy.deepcopy(args)

        with torch.enable_grad():
            y = DEQFunc.g(func, z1ss_temp, uss_temp, z0_temp, *args_temp)

        def g(x):
            y.backward(x, retain_graph=True)   # Retain for future calls to g
            JTx = z1ss_temp.grad.clone().detach()
            z1ss_temp.grad.zero_()
            return JTx + grad

        eps = 2e-10 * np.sqrt(bsz * seq_len * d_model)
        dl_df_est = torch.zeros_like(grad)

        result_info = broyden(g, dl_df_est, threshold=threshold, eps=eps, name="backward")
        dl_df_est = result_info['result']
        nstep = result_info['nstep']
        
        y.backward(torch.zeros_like(dl_df_est), retain_graph=False)

        grad_args = [None for _ in range(len(args))]
        return (None, dl_df_est, None, None, *grad_args)


class DEQForward(nn.Module):
    def __init__(self, func):
        """
        Initialize a module that computes the forward equilibrium of the input

        :param func: Transformation f_\theta that is applied on the input
        """
        super().__init__()
        self.func = func

    def _solve_equi(self, z1s, us, z0, **kwargs):
        """
        Solve for the equilibrium state in the forward pass

        :param z1s: The initial estimate of the equilibrium, of dimension (bsz x d_model x seq_len)
        :param us: Input tensor (or transformed input) to be injected (bsz x [-1] x (pad_len + seq_len))
        :param z0: History padding of dimension (bsz x d_model x pad_len)
        :param kwargs: Other arguments, such as positional embedding, training step information, etc.
        :return: The equilibrium state of dimension (bsz x d_model x seq_len)
        """
        raise NotImplemented

    def forward(self, z1s, us, z0, **kwargs):
        return self._solve_equi(z1s, us, z0, **kwargs)


class DEQBackward(nn.Module):
    def __init__(self, func, func_copy):
        """
        Initialize a module that, given the equilibrium state, prepares for the backward pass
        using the PyTorch autograd module (NOTE: There are better ways to implement the backward
        pass, but using func and func_copy makes it more flexible.)

        :param func: Transformation function f_\theta (whose parameters require gradient)
        :param func_copy: A copy of f_\theta whose parameters does NOT require gradient
        """
        super().__init__()
        self.func = func
        self.func_copy = func_copy

    def _solve_back(self, z1s, us, z0, **kwargs):
        """
        Registers the DEQ function f_\theta for backward updates via Thm. 1 in the paper.

        :param z1s: The equilibrium state output of dimension (bsz x d_model x seq_len)
        :param us: Input tensor (or transformed input) to be injected (bsz x [-1] x (pad_len + seq_len))
        :param z0: History padding of dimension (bsz x d_model x pad_len)
        :param kwargs: Other arguments, such as positional embedding, training step information, etc.
        :return: The same equilibrium state (but in practice, it is f_\theta(z*), which may be different from z*
        """
        raise NotImplemented

    def forward(self, z1s, us, z0, **kwargs):
        return self._solve_back(z1s, us, z0, **kwargs)