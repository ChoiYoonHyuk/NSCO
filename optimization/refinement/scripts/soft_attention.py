"""Soft attention and NumPy gradients."""
import numpy as np


def torch_values(V,q,W,scales,diagnostics=False):
    import torch
    keys=torch.einsum('bhni,mid->mbhnd',W,V)
    state=q[None].expand(len(V),-1,-1);values=[];saved=[]
    signs=torch.where(W[...,0]>=0,1.,-1.)
    for h in range(W.shape[1]):
        score=scales[:,h,None,None]*torch.einsum('mbnd,mbd->mbn',keys[:,:,h],state)
        weights=score.softmax(-1);values.append((weights*signs[None,:,h]).sum(-1))
        if diagnostics:saved.append((score,weights,state))
        state=(weights[...,None]*keys[:,:,h]).sum(-2)
    result=torch.stack(values,-1)
    return (result,saved)if diagnostics else result


def numpy_polynomial(values,coef,derivative=False):
    values=np.asarray(values);v=np.broadcast_to(coef,(len(values),len(coef))).astype(np.result_type(values,coef),copy=True)
    grad=np.zeros(v.shape+(values.shape[1],),dtype=v.dtype)if derivative else None
    for h in range(values.shape[1]):
        if derivative:
            nxt=grad[:,::2]+values[:,h,None,None]*grad[:,1::2];nxt[:,:,h]+=v[:,1::2];grad=nxt
        v=v[:,::2]+values[:,h,None]*v[:,1::2]
    return(v[:,0],grad[:,0])if derivative else v[:,0]


def numpy_attention(A,q,W,Y,coef,scales):
    V=np.vstack((np.zeros((1,A.shape[1]),dtype=A.dtype),A))
    keys=W@V;state=q;values=[];cache=[]
    for h in range(W.shape[1]):
        key=keys[:,h];scores=scales[h]*np.einsum('bnd,bd->bn',key,state)
        weight=np.exp(scores-scores.real.max(1,keepdims=True));weight/=weight.sum(1,keepdims=True)
        signs=np.where(W[:,h,:,0]>=0,1,-1);values.append((weight*signs).sum(1));cache.append((key,state,weight,signs))
        state=(weight[:,:,None]*key).sum(1)
    values=np.stack(values,1);out,derivative=numpy_polynomial(values,coef,True)
    dv=(out-Y)[:,None]*derivative/len(q);total=np.zeros_like(V);dstate=np.zeros((len(q),A.shape[1]))
    for h in reversed(range(W.shape[1])):
        key,state,weight,signs=cache[h]
        dweight=dv[:,h,None]*signs+np.einsum('bnd,bd->bn',key,dstate)
        dscore=weight*(dweight-(weight*dweight).sum(1,keepdims=True))
        dkey=weight[:,:,None]*dstate[:,None,:]+scales[h]*dscore[:,:,None]*state[:,None,:]
        total+=np.einsum('bni,bnj->ij',W[:,h],dkey)
        dstate=scales[h]*np.einsum('bn,bnd->bd',dscore,key)
    return float(.5*np.mean((out-Y)**2)),total[1:],values,out
