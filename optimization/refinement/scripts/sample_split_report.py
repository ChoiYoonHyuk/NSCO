"""Complete fixed-family inference; no contrast or condition is selected away."""
from pathlib import Path
import json,math,hashlib
import numpy as np
from scipy.stats import t
R=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=json.loads((R/'sample_split_protocol.json').read_text());f=R/'results/split_confirmation/raw_results.json';rows=json.loads(f.read_text());summ=[];tests=[]
 for case in p['cases']:
  for g in case['readouts']:
   values={}
   for family in p['families']:
    rr=[x for x in rows if x['case']==case['name']and x['readout']==g and x['family']==family]
    rr=sorted(rr,key=lambda x:x['seed']);assert[x['seed']for x in rr]==p['confirmation_seeds']
    v=np.array([x['hard_error']for x in rr]);values[family]=v
    for metric in ['hard_error','calibrated_risk','frame_error','subspace_squared']:
     z=np.array([x[metric]for x in rr]);h=float(t.ppf(.975,len(z)-1)*z.std(ddof=1)/math.sqrt(len(z)))
     summ.append({'case':case['name'],'readout':g,'family':family,'metric':metric,'mean':float(z.mean()),'ci95_halfwidth':h,'seed_values':z.tolist(),'rate':rr[0]['rate'],'alpha':rr[0]['alpha'],'primary':g in case['primary_readouts']and metric=='hard_error'})
   if g in case['primary_readouts']:
    for control in ['sgd','sgd_blend','plain_shrink','pilot']:
     z=values['cv_shrink']-values[control];n=len(z);sgn=1-2*((np.arange(2**n)[:,None]>>np.arange(n))&1)
     pv=float(np.mean(np.abs(sgn@z/n)>=abs(z.mean())-1e-14));half=float(t.ppf(.975,n-1)*z.std(ddof=1)/math.sqrt(n))
     tests.append({'case':case['name'],'readout':g,'candidate':'cv_shrink','control':control,'difference':float(z.mean()),'ci95_low':float(z.mean()-half),'ci95_high':float(z.mean()+half),'p_two_sided':pv,'paired_seed_differences':z.tolist()})
 assert len(tests)==12
 order=np.argsort([x['p_two_sided']for x in tests]);prev=0.
 for rank,i in enumerate(order):
  prev=max(prev,min(1.,(len(order)-rank)*tests[i]['p_two_sided']));tests[i]['p_holm']=prev
  tests[i]['decision']='favor_candidate'if prev<.05 and tests[i]['difference']<0 else('favor_control'if prev<.05 and tests[i]['difference']>0 else'inconclusive')
 counts={x:sum(z['decision']==x for z in tests)for x in ['favor_candidate','favor_control','inconclusive']}
 out={'status':'complete','summary':summ,'tests':tests,'decision_counts':counts,'input_sha256':{'sample_split_protocol.json':sha(R/'sample_split_protocol.json'),'results/split_confirmation/raw_results.json':sha(f),'fixed_split_choices.json':sha(R/'fixed_split_choices.json')},'scope':'Twelve prespecified hard-error paired sign-flip comparisons, assuming symmetry of paired differences, with one Holm family. Nominal t15 intervals are not simultaneous. MP6 and all native soft risks are descriptive. Calibrated risk equals2lambda²hard error and is not independent soft improvement. No universal optimizer-superiority claim.'}
 path=R/'results/split_confirmation/report.json';assert not path.exists();path.write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8');print(json.dumps({'status':'complete','counts':counts,'tests':tests}))
if __name__=='__main__':main()
