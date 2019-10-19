import torch


def neumann(JfT, v0, threshold, eps):
    vt = v0
    gt = v0
    nstep = 0
    for t in range(threshold):
        vt = JfT(vt)
        gt += vt
        nstep += 1
        if torch.norm(vt).item() <= eps:
            break
    return {'result': gt,
            'nstep': nstep,
            'diff': torch.norm(vt).item()}