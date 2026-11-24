"""Shared Gaussian and deployment kernels."""
import numpy as np


def sign(x):return np.where(x>=0,1,-1)


def qr(rng,D,d):
    u,r=np.linalg.qr(rng.standard_normal((D,d)));return u*np.where(np.diag(r)>=0,1.,-1.)[None,:]


def hard(q,W,P):
    keys=W@P;query=q;ii=[];vv=[];rows=np.arange(len(q))
    for h in range(W.shape[1]):
        i=(keys[:,h]*query[:,None,:]).sum(-1).argmax(1);ii.append(i);vv.append(sign(W[rows,h,i,0]));query=keys[rows,h,i]
    return np.stack(ii,1),np.stack(vv,1)


def configure_torch(device):
    import torch
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(110011);return {'torch':torch.__version__,'device':device,'device_name':torch.cuda.get_device_name() if device.startswith('cuda') else 'CPU',
        'numpy':np.__version__,'python':__import__('platform').python_version(),'dtype':'float64','tf32':False,'deterministic_algorithms':True}
