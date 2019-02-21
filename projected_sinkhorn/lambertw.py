# Reimplementation of scipy version of lambertw for branching factor = 0
import torch
import math
import warnings

OMEGA = 0.56714329040978387299997  # W(1, 0)
EXPN1 = 0.36787944117144232159553  # exp(-1)

def evalpoly(coeff, degree, z): 
    powers = torch.arange(degree,-1,-1).float().to(z.device)
    return ((z.unsqueeze(-1)**powers)*coeff).sum(-1)

def lambertw(z0, tol=1e-5): 
    # this is a direct port of the scipy version for the 
    # k=0 branch for *positive* z0 (z0 >= 0)

    # skip handling of nans
    if torch.isnan(z0).any(): 
        raise NotImplementedError

    w0 = z0.new(*z0.size())
    # under the assumption that z0 >= 0, then I_branchpt 
    # is never used. 
    I_branchpt = torch.abs(z0 + EXPN1) < 0.3
    I_pade0 = (-1.0 < z0)*(z0 < 1.5)
    I_asy = ~(I_branchpt | I_pade0)
    if I_pade0.any(): 
        z = z0[I_pade0]
        num = torch.Tensor([
            12.85106382978723404255,
            12.34042553191489361902,
            1.0
        ]).to(z.device)
        denom = torch.Tensor([
            32.53191489361702127660,
            14.34042553191489361702,
            1.0
        ]).to(z.device)
        w0[I_pade0] = z*evalpoly(num,2,z)/evalpoly(denom,2,z)

    if I_asy.any(): 
        z = z0[I_asy]
        w = torch.log(z)
        w0[I_asy] = w - torch.log(w)

    # split on positive and negative, 
    # and ignore the divergent series case (z=1)
    w0[z0 == 1] = OMEGA
    I_pos = (w0 >= 0)*(z0 != 1)
    I_neg = (w0 < 0)*(z0 != 1)
    if I_pos.any(): 
        w = w0[I_pos]
        z = z0[I_pos]
        for i in range(100): 
            # positive case
            ew = torch.exp(-w)
            wewz = w - z*ew
            wn = w - wewz/(w + 1 - (w + 2)*wewz/(2*w + 2))

            if (torch.abs(wn - w) < tol*torch.abs(wn)).all():
                break
            else:
                w = wn
        w0[I_pos] = w

    if I_neg.any(): 
        w = w0[I_neg]
        z = z0[I_neg]
        for i in range(100):
            ew = torch.exp(w)
            wew = w*ew
            wewz = wew - z
            wn = w - wewz/(wew + ew - (w + 2)*wewz/(2*w + 2))
            if (torch.abs(wn - w) < tol*torch.abs(wn)).all():
                break
            else:
                w = wn
        w0[I_neg] = wn
    return w0

if __name__ == '__main__': 
    from scipy.special import lambertw as sp_lamw
    import numpy as np

    torch.random.manual_seed(0)
    x = torch.randn(1000).abs()
    
    torch_lamw = lambertw(x)
    scipy_lamw = torch.from_numpy(np.real(sp_lamw(x.numpy()))).float()
    # print(torch_lamw[:10], scipy_lamw[:10])
    print((torch_lamw - scipy_lamw).abs().max())
    print(lambertw(torch.ones(1)*1e-8), sp_lamw(1e-8))
