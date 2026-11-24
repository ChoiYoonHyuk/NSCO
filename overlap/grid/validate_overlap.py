"""Outcome-independent identities and finite-Monte-Carlo smoke checks."""
from run_overlap import *
def main():
    g=rng(2026112099,1);checks=[]
    for d in [2,4]:
        base=polar(g.standard_normal((15,d)));xx,ww=quadrature(d)
        for rho in [0.,.5,1.]:
            P=target(base,rho);_,hp,ho=scales(rho,xx,ww,d)
            expected=P[1:]*np.array([hp]+[ho]*(d-1))[None]
            maxerr=float(np.max(np.abs(recover(.123*expected,rho,d)-P)))
            assert maxerr<1e-12,maxerr
            n=65536;q,W=draw(g,n,1,16,d);_,v=path(q,W,P)
            delta=(W[:,0,0]-W[:,0,1])/math.sqrt(2);R=delta[:,0,None]*delta[:,1:]
            obs=v[:,0,None,None]*R[:,:,None]*q[:,None,:]
            z=(obs.mean(0)-expected)/(obs.std(0,ddof=1)/math.sqrt(n))
            maxz=float(np.max(np.abs(z)));assert maxz<6,maxz
            checks.append({'d':d,'rho':rho,'population_recovery_max_abs':maxerr,'first_moment_max_standard_errors':maxz,'examples':n})
    p=json.loads((ROOT/'protocol.json').read_text());cs=coefficient_study({**p,'coefficient_examples':16384},ROOT/'smoke')
    q,W=draw(g,32768,7,16,4)
    c0=next(c for c in cs if c['d']==4 and c['rho']==0)
    gg=gates(W,7,c0);equality=max(float(np.max(np.abs(x-gg[0]))) for x in gg)
    assert equality==0
    prob=1-math.erf(1.)**2;norm=math.sqrt(.5*prob*(1-prob))
    values=g.standard_normal((262144,2));b=sign(values).mean(1);psi=(b==0)*((np.abs(values).sum(1)>=2)-prob)/norm
    magchecks={'mean':float(psi.mean()),'second_moment':float(np.mean(psi*psi)),'corr_with_b':float(np.mean(psi*b)),
               'corr_with_opposite_sign_indicator':float(np.mean(psi*(b==0)))}
    assert abs(magchecks['mean'])<6/math.sqrt(len(psi))
    assert abs(magchecks['second_moment']-1)<6*np.std(psi*psi)/math.sqrt(len(psi))
    exact=filters.exact_checks(5)
    write(ROOT/'smoke'/'validation.json',{'status':'passed','first_moment_checks':checks,'zero_overlap_filter_max_difference':equality,
       'magnitude_orthogonality':magchecks,'exact_filter_checks':exact})
    print('All overlap validation checks passed')
if __name__=='__main__':main()
