"""Aggregate all registered seeds; audit stored endpoint frames by reconstructing full ambient keys."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='2'
import csv,json,math,itertools,hashlib,time
from pathlib import Path
import numpy as np
from scipy.stats import t
from depth import ROOT,rowspec,test_data,basis,gate
from run_overlap import target,path,write,METHODS
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

O=ROOT/'production'
p=json.loads((O/'registration.json').read_text())['p']
cs={c['rho']:c for c in json.loads((O/'coefficients.json').read_text())}
cap=p['budgets'][-1];seeds=p['seeds'];spec=list(rowspec(p))
def writecsv(name,data):
    with (ROOT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
def avg(values):
    return float(np.mean(values))
def ci(x):
    x=np.asarray(x);bar=float(x.mean());w=float(t.ppf(.975,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x)))
    return (bar,bar-w,bar+w)
def digest(f):return hashlib.sha256(f.read_bytes()).hexdigest()
def rate(x):
    return sum(a is not None for a in x)
def estimate(x):
    ordered=sorted(cap if a is None else a for a in x)
    return float(np.median(ordered)),rate(x)<len(x)//2+1
def display(x):
    m,b=estimate(x);v=math.log2(m)
    return ('>' if b else '')+('%g'%v if abs(round(v)-v)<1e-8 else f'{v:.2f}')
def getkey(D,r,k,m,setting='exact'):
    return (D,r,k,m,setting)
def zeromass(k,m,rfit):
    C=cs[rfit]; states=np.asarray(list(itertools.product(range(4),repeat=k-1)))
    # Realize each sign/magnitude class by two representative public values.
    gvals=np.array([[-1.,-1.],[1.,1.],[-.25,.25],[-2.,2.]])
    bvals=np.array([-1.,1.,0.,0.]);q=C['magnitude']['tail_probability']
    weights=np.prod(np.array([.25,.25,.5*(1-q),.5*q])[states],axis=1)
    g=np.zeros((len(states),k,2));g[:,1:]=gvals[states]
    f=gate(bvals[states],g,k,m,C)
    return float(weights[f==0].sum())
def summarize():
    completion=json.loads((O/'completion.json').read_text())
    groups={};end={};observed=0
    for seed in seeds:
        file=O/f'{seed}.csv'
        assert digest(file)==completion['results'][file.name]
        with file.open() as f:
            for R in csv.DictReader(f):
                A={x:(v if x in ('method','setting') or v=='' else float(v)) for x,v in R.items()}
                A['gate_signal_variance']=A['gate_second_moment']-A['gate_identifying_signal']**2
                A['gate_signal_to_variance']=A['gate_identifying_signal']**2/A['gate_signal_variance']
                D,r,k,m,setting,n=(A[x] for x in ('D','rho','k','method','setting','n'))
                key=(D,r,k,m,setting);seq=groups.setdefault(key,{}).setdefault(seed,[])
                seq.append((int(n),A['clean_error'],A['frame_error_normalized']))
                if n==cap:end.setdefault(key,{})[seed]=A
                observed+=1
    assert observed==completion['rows']==len(seeds)*len(spec)*len(p['ambient'])*len(p['budgets'])
    for key,sgroup in groups.items():
        assert set(sgroup)==set(seeds)
        for s,results in sgroup.items():assert [v[0] for v in results]==p['budgets']
    metrics=['clean_error','frame_error_normalized','complete_chain_accuracy','public_values_only_error','gate_identifying_signal','gate_second_moment','gate_snr','gate_signal_variance','gate_signal_to_variance','gate_seconds','moment_seconds','recovery_seconds','coefficient_seconds','public_mc_examples']+['hop'+str(h) for h in range(1,10)]
    cache={}; summary={}; crossings={};cr=[];ss=[]
    for key,sg in groups.items():
        D,r,k,m,z=key;base=dict(D=int(D),rho=r,k=int(k),method=m,setting=z)
        a=list(end[key].values())[0];u=(int(k),m,a['supplied_norm'])
        if u not in cache:cache[u]=zeromass(*u)
        row=dict(**base,nmax=cap,zero_weight_probability=cache[u],total_labels_per_run=cap)
        for met in metrics:
            vals=[v[met] for v in end[key].values() if v[met]!='']
            vals=ci(vals) if vals else ('','','')
            for suf,val in zip(['','_lo','_hi'],vals):row[met+suf]=val
        for i,name,eps in [(1,'hard',p['prediction_threshold']),(2,'frame',p['frame_threshold'])]:
            obs=[]
            for s in seeds:
                hit=next((v[0] for v in sg[s] if v[i]<=eps),None);obs.append(hit)
                cr.append(dict(**base,seed=s,endpoint=name,threshold=eps,n_first=hit or '',
                               right_censored=int(hit is None),censor_at=cap if hit is None else ''))
            crossings[key,name]=obs
            med,bound=estimate(obs);row[name+'_median_or_lower']=med;row[name+'_lower_bound']=int(bound);row[name+'_success']=rate(obs)
        ss.append(row); summary[key]=row
    writecsv('crossings.csv',cr);writecsv('cell_summary.csv',ss)
    contrasts=[]
    def save(D,r,k,name,metric,values,kind):
        mu,lo,hi=ci(values)
        contrasts.append(dict(D=D,rho=r,k=k,contrast=name,metric=metric,estimand=kind,n_seeds=len(values),
                              mean=mu,ci95_lo=lo,ci95_hi=hi,seed_differences=' '.join('%.12g'%v for v in values)))
    for D in p['ambient']:
        for r in p['overlaps']:
            for k in p['depths']:
                ex=(D,r,k,METHODS[-1],'exact')
                alternatives=[(D,r,k,m,'exact') for m in METHODS[:-1]]
                if k in p['perturbed_depths'] and r>0: alternatives += [(D,r,k,METHODS[-1],z) for z in ['direction','magnitude']]
                for b in alternatives:
                    label='exact sign_magnitude minus '+b[3]+' '+b[4]
                    for metric in metrics[:7]:
                        d=[end[ex][s][metric]-end[b][s][metric] for s in seeds]
                        save(D,r,k,label,metric,d,'endpoint paired difference')
                    for metric in ['hard','frame']:
                        a1=crossings[ex,metric];b1=crossings[b,metric]
                        d=[math.log2((v or cap)/(w or cap)) for v,w in zip(a1,b1)]
                        save(D,r,k,label,metric,d,'capped log2 first grid budget')
            for metric in ['hard','frame']:
                for m in METHODS:
                    a1=crossings[(D,r,9,m,'exact'),metric];b1=crossings[(D,r,3,m,'exact'),metric]
                    d=[math.log2((v or cap)/(w or cap))/6 for v,w in zip(a1,b1)]
                    save(D,r,'3-9',m,metric,d,'mean capped log2 cost increase per hop')
        for m in METHODS:
            for metric in ['hard','frame']:
                for r in [.5,1.]:
                    def growth(rho):
                        a=crossings[(D,rho,9,m,'exact'),metric]
                        b=crossings[(D,rho,3,m,'exact'),metric]
                        return np.array([math.log2((v or cap)/(w or cap))/6 for v,w in zip(a,b)])
                    delta=growth(r)-growth(0.)
                    save(D,r,'3-9',m+' rho minus zero overlap',metric,delta,
                         'paired change in capped log2 cost growth per hop')
    writecsv('paired.csv',contrasts)
    return summary,crossings,end,observed

def audit(end):
    from depth import sha
    lock=json.loads((O/'registration.json').read_text())['source_sha256']
    assert all(sha(ROOT/f)==h for f,h in lock.items())
    count=0;maxdiff=0
    for seed in seeds:
        q,g,X,truths=test_data(p,seed);frames=np.load(O/f'frames_{seed}.npz')
        for D in p['ambient']:
            B=basis(seed,D);W=np.concatenate([g[...,None],X[...,:D-1]@B.T],axis=3)
            for r in p['overlaps']:
                idx,v=path(q,W,target(B[:,:2],r));ii,vv=truths[r]
                assert np.array_equal(idx,ii) and np.array_equal(v,vv)
            for j,(r,k,m,z,rf,deg) in enumerate(spec):
                A=end[D,r,k,m,z][seed];F=frames[f's{j}D{D}'];ii,vv=truths[r]
                ix,v=path(q,W[:,:k],F);eq=ix==ii[:,:k]
                checks=[(A['clean_error'],np.mean(v.prod(1)!=vv[:,:k].prod(1))),
                        (A['frame_error_normalized'],np.linalg.norm(F-target(B[:,:2],r))/math.sqrt(2)),
                        (A['complete_chain_accuracy'],eq.all(1).mean())]
                checks += [(A['hop'+str(h+1)],eq[:,h].mean()) for h in range(k)]
                maxdiff=max(maxdiff,max(abs(a-b) for a,b in checks));count+=1
                for f in ['clean_error','complete_chain_accuracy','frame_error_normalized','gate_snr']:
                    if r==0:assert A[f]==end[D,0,k,METHODS[0],'exact'][seed][f]
                assert A['final_labels']==cap and A['additional_supervision_labels']==0
                assert abs(np.linalg.norm(F[0])-rf)<1e-12 and np.linalg.norm(F.T@F-np.eye(2))<1e-10
    assert maxdiff<1e-11
    return dict(checked_frames=count,independent_endpoint_max_difference=maxdiff,source_sha256_verified=lock,
                zero_overlap_filters_identical=True,teacher_reconstructed_from_full_ambient_X=True,
                all_label_budgets_charge_zero_weight_examples=True)

def plot(summary,cross):
    colors=['#3c4856','#d87710','#2672a0','#782876'];marks=['o','s','D','^']
    fig,axes=plt.subplots(2,3,figsize=(9,4.9),sharex=True,sharey=True)
    for i,D in enumerate(p['ambient']):
        for j,r in enumerate(p['overlaps']):
            ax=axes[i,j];ax.set_title('D = %d,  ρ = %g'%(D,r),fontsize=10)
            for mi,m in enumerate(METHODS):
                x=np.array(p['depths'])+(mi-1.5)*.12
                for style,metric in [('solid','hard'),('dotted','frame')]:
                    ys=[];bounds=[]
                    for k in p['depths']:
                        v,b=estimate(cross[(D,r,k,m,'exact'),metric]);ys.append(math.log2(v));bounds.append(b)
                    ax.plot(x,np.where(bounds,np.nan,ys),color=colors[mi],ls=style,lw=1.25)
                    for t,(xx,y,b) in enumerate(zip(x,ys,bounds)):
                        if b:ax.annotate('',xy=(xx,y+.25),xytext=(xx,y-.13),arrowprops=dict(arrowstyle='->',lw=.8,color=colors[mi]))
                        else:ax.plot(xx,y,marker=marks[mi],ls='',markersize=3.5,markeredgewidth=.8,color=colors[mi],markerfacecolor='white' if metric=='frame' else colors[mi])
            ax.grid(axis='y',alpha=.25);ax.set_xticks([3,5,7,9]);ax.set_ylim(11.6,21.7);ax.set_xlim(2.6,9.5)
            if j==0:ax.set_ylabel('log₂ median labels')
            if i==1:ax.set_xlabel('Depth k')
    handles=[mlines.Line2D([],[],marker=marks[i],color=colors[i],linestyle='',label=x) for i,x in enumerate(['Certificate','Adjacent pair','Complete sign','Sign + magnitude'])]
    handles += [mlines.Line2D([],[],color='black',ls=s,label=l) for s,l in [('solid','Hard error ≤ 1/8'),('dotted','Frame error ≤ .35')]]
    fig.legend(handles=handles,loc='lower center',ncol=3,fontsize=8,frameon=False,bbox_to_anchor=(.5,.003))
    fig.tight_layout(rect=[0,.10,1,1],pad=.75)
    fig.savefig(ROOT/'depth_scaling.pdf',bbox_inches='tight');fig.savefig(ROOT/'depth_scaling.png',dpi=185,bbox_inches='tight');plt.close(fig)
if __name__ == '__main__':
    S,C,E,obs=summarize()
    plot(S,C)
    a=audit(E)
    write(ROOT/'audit.json', dict(rows_verified=obs, **a))
