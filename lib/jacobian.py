import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


def jac_loss_estimate(f0, z0, vecs=2, create_graph=True):
    """Estimating tr(J^TJ)=tr(JJ^T) via Hutchinson estimator

    Args:
        f0 (torch.Tensor): Output of the function f (whose J is to be analyzed)
        z0 (torch.Tensor): Input to the function f
        vecs (int, optional): Number of random Gaussian vectors to use. Defaults to 2.
        create_graph (bool, optional): Whether to create backward graph (e.g., to train on this loss). 
                                       Defaults to True.

    Returns:
        torch.Tensor: A 1x1 torch tensor that encodes the (shape-normalized) jacobian loss
    """
    vecs = vecs
    result = 0
    for i in range(vecs):
        v = torch.randn(*z0.shape).to(z0)
        vJ = torch.autograd.grad(f0, z0, v, retain_graph=True, create_graph=create_graph)[0]
        result += vJ.norm()**2
    return result / vecs / np.prod(z0.shape)

def power_method(f0, z0, n_iters=200):
    """Estimating the spectral radius of J using power method

    Args:
        f0 (torch.Tensor): Output of the function f (whose J is to be analyzed)
        z0 (torch.Tensor): Input to the function f
        n_iters (int, optional): Number of power method iterations. Defaults to 200.

    Returns:
        tuple: (largest eigenvector, largest (abs.) eigenvalue)
    """
    evector = torch.randn_like(z0)
    bsz = evector.shape[0]
    for i in range(n_iters):
        vTJ = torch.autograd.grad(f0, z0, evector, retain_graph=(i < n_iters-1), create_graph=False)[0]
        evalue = (vTJ * evector).reshape(bsz, -1).sum(1, keepdim=True) / (evector * evector).reshape(bsz, -1).sum(1, keepdim=True)
        evector = (vTJ.reshape(bsz, -1) / vTJ.reshape(bsz, -1).norm(dim=1, keepdim=True)).reshape_as(z0)
    return (evector, torch.abs(evalue))