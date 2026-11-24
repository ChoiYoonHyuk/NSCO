"""Post-execution summaries; does not train, select models, or change locked sources."""
from pathlib import Path
import json,math,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parent
P=json.loads((ROOT/'protocol.json').read_text())
T={7:2.3646242510102993,11:2.200985160082949}
def stats(v):
    a=np.asarray(v,dtype=float);n=len(a);m=float(a.mean());h=T[n-1]*float(a.std(ddof=1))/math.sqrt(n)
    return {'mean':m,'ci95_half_width':h,'ci95':[m-h,m+h],'n_seeds':n,'seed_values':a.tolist()}
def load(s):return json.loads((ROOT/'results'/s/'rows.json').read_text())
def crossing(rows,threshold):
    rows=sorted(rows,key=lambda x:x['n'])
    for j,r in enumerate(rows):
        if r['frame_error']<=threshold:
            if j==0:return {'n':r['n'],'left_censored':True,'right_censored':False,'bracket':[None,r['n']]}
            prev=rows[j-1];u=(math.log(threshold)-math.log(prev['frame_error']))/(math.log(r['frame_error'])-math.log(prev['frame_error']))
            return {'n':math.exp(math.log(prev['n'])+u*math.log(r['n']/prev['n'])),'left_censored':False,'right_censored':False,'bracket':[prev['n'],r['n']]}
    return {'n':None,'right_censored':True,'left_censored':False,'bracket':[rows[-1]['n'],None]}
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
result={'protocol_sha256':hashlib.sha256((ROOT/'protocol.json').read_bytes()).hexdigest(),'interval_note':'Pointwise 95% t intervals across independent seeds; no multiple-testing-adjusted significance claims. Test-sample binomial uncertainty is not the seed interval.'}
r=load('same_beta');cuts=[];logratios=[]
for s in P['same_beta']['seeds']:
    a=crossing([x for x in r if x['seed']==s and x['readout']=='A'],.35);b=crossing([x for x in r if x['seed']==s and x['readout']=='B'],.35)
    cuts.append({'seed':s,'A':a,'B':b,'ratio':b['n']/a['n'] if a['n'] and b['n'] else None})
    if a['n'] and b['n']:logratios.append(math.log(b['n']/a['n']))
assert len(logratios)==len(cuts),'Do not silently drop censored thresholds'
rr=np.random.default_rng(73777);lr=np.array(logratios);boot=np.exp(lr[rr.integers(0,len(lr),(10000,len(lr)))].mean(1))
curves=[]
for n in P['same_beta']['budgets']:
    for name in ['A','B']:
        group=[x for x in r if x['n']==n and x['readout']==name]
        curves.append({'n':n,'readout':name,**{m:stats([x[m] for x in group]) for m in ['frame_error','clean_error','empirical_derivative_mass']}})
result['same_beta']={'curves':curves,'crossings':cuts,'ratio_geometric_mean':float(np.exp(lr.mean())),'ratio_bootstrap_ci95':np.quantile(boot,[.025,.975]).tolist(),'right_censored':sum(x['A']['right_censored'] or x['B']['right_censored'] for x in cuts),'threshold_interpolation':'First observed crossing, linear interpolation of log frame error against log label count between the adjacent budgets; every seed retained.', 'predicted_inverse_mass_ratio':2.6}
for study in ['position','sgd']:
    r=load(study);groups={}
    for x in r:
        key=(x['n'],x['readout'],x['method'],x.get('deployment','polar' if x['frame_error'] is not None else 'public_mean_unidentified'))
        groups.setdefault(key,[]).append(x)
    summary=[]
    for (n,readout,method,deployment),gg in sorted(groups.items()):
        row={'n':n,'readout':readout,'method':method,'deployment':deployment}
        for metric in ['clean_error','frame_error','projector_error','soft_error','soft_excess_risk','literal_zero_hard_argmax_error','complete_path_accuracy']:
            vals=[x[metric] for x in gg if x.get(metric) is not None]
            if vals:row[metric]=stats(vals)
        if gg[0].get('hop_accuracy') is not None:row['hop_accuracy']=[stats([x['hop_accuracy'][j] for x in gg]) for j in range(P[study]['k'])]
        summary.append(row)
    result[study]={'curves':summary,'labels_per_seed_at_endpoint':max(P[study]['budgets']),'test_inputs_per_seed':P['test_examples']}
    if study=='sgd':result[study]['terminal_zero_gradient_all_exact']=all(x['terminal_zero_gradient_max']==0 for x in r)
result['run_seconds']={s:json.loads((ROOT/'results'/s/'manifest.json').read_text())['seconds'] for s in ['same_beta','position','sgd']}
write(ROOT/'summary.json',result)
print(json.dumps({'same_beta_ratio':result['same_beta']['ratio_geometric_mean'],'ratio_ci':result['same_beta']['ratio_bootstrap_ci95'],'endpoint_position':[x for x in result['position']['curves'] if x['n']==262144],'endpoint_sgd':[{'readout':x['readout'],'method':x['method'],'deployment':x['deployment'],'error':x['clean_error']['mean']} for x in result['sgd']['curves'] if x['n']==262144]},indent=2))
