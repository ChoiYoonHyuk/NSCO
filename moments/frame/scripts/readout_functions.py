"""Public exact Boolean kernels for the fixed empirical sample-scaling study."""
from __future__ import annotations
import itertools
import math
import numpy as np
from gaussian_model import ROOT, digest, generator, write_json


def bits(k):
    return 2*((np.arange(1<<k)[:,None] >> np.arange(k))&1)-1


def walsh(values):
    a=np.asarray(values,dtype=float).copy()
    h=1
    while h<len(a):
        for j in range(0,len(a),2*h):
            x=a[j:j+h].copy();y=a[j+h:j+2*h].copy()
            a[j:j+h]=x+y;a[j+h:j+2*h]=x-y
        h*=2
    # Transform convention uses (-1)^bit; our bits are 2*bit-1.
    return a/len(a)*np.array([(-1.)**i.bit_count() for i in range(len(a))])


def extension_grid(values,k,nodes=(-1.,0.,1.)):
    """Ternary cube table, coordinate zero is fastest, ternary digit 0/1/2=-1/0/+1."""
    t=np.asarray(values,dtype=float).reshape((2,)*k,order='F')
    for axis in range(k):
        lo=np.take(t,0,axis=axis);hi=np.take(t,1,axis=axis)
        t=np.stack(tuple((1-x)/2*lo+(1+x)/2*hi for x in nodes),axis=axis)
    return t.reshape(-1,order='F')


def catalogue(protocol):
    specs=[]
    for r in protocol['readouts']:
        family,k=r['family'],r['depth'];v=bits(k)
        if family=='parity': y=v.prod(1)
        elif family=='majority_product': y=np.sign(v.reshape(-1,k//3,3).sum(2)).prod(1)
        elif family=='odd_majority': y=np.sign(v.sum(1))
        elif family=='random_balanced':
            y=np.random.default_rng(r['truth_seed']).permutation(np.r_[-np.ones(1<<(k-1)),np.ones(1<<(k-1))])
        else: raise ValueError(family)
        c=walsh(y);deg=np.array([m.bit_count() for m in range(1<<k)])
        beta=float(np.sum(c*c*2.**-deg));kap=float(np.sum(c[1::2]**2*2.**(1-deg[1::2])))
        derivative=extension_grid((y[1::2]-y[0::2])/2,k-1)
        specs.append({**r,'id':family+'_k'+str(k),'truth':y,'fourier':c,'derivative_table':derivative,
                      'beta':beta,'kappa1':kap,'b_beta':-math.log2(beta),'b_kappa1':-math.log2(kap)})
    annotations=protocol['downstream_annotations']
    for s in tuple(specs):
        if s['id'] not in annotations['readouts']:continue
        k=s['depth'];mu=annotations['noise_contrast']
        table=extension_grid((s['truth'][1::2]-s['truth'][0::2])/2,k-1,(-1.,-mu,0.,mu,1.))
        for pi in annotations['probabilities']:
            r=(1+pi*mu*mu)/2
            kap=float(sum(s['fourier'][mask]**2*r**(mask.bit_count()-1) for mask in range(1,1<<k,2)))
            specs.append({**s,'id':s['id']+'_pi'+str(pi).replace('.','_'),'annotation_probability':pi,
                          'posterior_second_moment':r,'derivative_table':table,'kappa1':kap,'b_kappa1':-math.log2(kap)})
    return specs


def label(v,spec):
    k=spec['depth'];ix=(((v[:,:k]+1)//2).astype(np.int64)*(1<<np.arange(k))).sum(1)
    return spec['truth'][ix]


def derivative(g,spec,annotation_uniform=None,annotation_labels=None):
    k=spec['depth'];b=np.where(g[:,1:k]>=0,1,-1).mean(2).astype(np.int64)
    if 'annotation_probability' in spec:
        digits=2*(b+1)
        selected=(b==0)&(annotation_uniform[:,:k-1]<spec['annotation_probability'])
        digits=np.where(selected,2+annotation_labels[:,:k-1],digits)
        ix=(digits*5**np.arange(k-1)).sum(1)
    else:ix=((b+1)*3**np.arange(k-1)).sum(1)
    return spec['derivative_table'][ix]


def self_test(protocol):
    specs=catalogue(protocol);rng=np.random.default_rng(77891);checks=[]
    for spec in specs:
        k=spec['depth'];v=bits(k);c=spec['fourier'];y=spec['truth']
        reconstructed=np.zeros(len(v))
        for mask,coef in enumerate(c):
            reconstructed+=coef*v[:,[j for j in range(k) if mask>>j&1]].prod(1)
        assert np.array_equal(reconstructed,y)
        assert y.mean()==0 and np.isclose(np.sum(c*c),1)
        # Independent direct polynomial derivative on random ternary coordinates.
        z=rng.standard_normal((97,k,2));b=np.where(z>=0,1,-1).mean(2)
        expected=np.zeros(len(z));u=rng.random((len(z),k-1));ann=rng.choice([-1,1],size=(len(z),k-1))
        if 'annotation_probability' in spec:
            mask=u<spec['annotation_probability'];mu=protocol['mu']
            b[:,1:]=np.where(mask,(b[:,1:]+mu*ann)/(1+mu*ann*b[:,1:]),b[:,1:])
        for mask,coef in enumerate(c):
            if mask&1: expected+=coef*b[:,[j for j in range(1,k) if mask>>j&1]].prod(1)
        assert np.allclose(derivative(z,spec,u,ann),expected,atol=1e-14)
        # Enumerate all ternary values with exact product probabilities for E[D1^2].
        base=5 if 'annotation_probability' in spec else 3
        digits=np.array(list(itertools.product(range(base),repeat=k-1)))
        ix=(digits*base**np.arange(k-1)).sum(1)
        if base==3: weights=np.prod(np.where(digits==1,.5,.25),1)
        else:
            pi=spec['annotation_probability'];w=np.array([.25,pi/4,(1-pi)/2,pi/4,.25]);weights=np.prod(w[digits],1)
        assert np.isclose(np.sum(weights*spec['derivative_table'][ix]**2),spec['kappa1'])
        if spec['family']=='parity' and base==3: assert spec['beta']==2.**-k and spec['kappa1']==2.**(1-k)
        if spec['family']=='majority_product' and base==3:
            assert np.isclose(spec['beta'],(13/32)**(k//3))
            assert np.isclose(spec['kappa1'],(10/13)*(13/32)**(k//3))
        checks.append(spec['id'])
    return {'truth_fourier_reconstruction':checks,'public_derivative_exact':checks,'derivative_variance_exact':checks}
