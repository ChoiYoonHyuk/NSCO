"""Fixed averaged-SGD confirmation summaries and the nine prespecified paired tests.

No setting is selected here. Every completed confirmation run is included.
The exact sign-flip tests require sign-symmetric paired run differences.
"""
from pathlib import Path
import hashlib
import json
from collections import Counter
import numpy as np
from scipy.stats import t

R=Path(__file__).resolve().parents[1]
FAMILIES=['averaged_sgd','endpoint_sgd','avg_plain','avg_online','avg_meancontrol','gated']


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summary(values):
    x=np.asarray(values,dtype=float)
    assert np.isfinite(x).all() and len(x)==16
    mean=float(x.mean());se=float(x.std(ddof=1)/np.sqrt(len(x)));hw=float(t.ppf(.975,len(x)-1)*se)
    return {'n':len(x),'mean':mean,'sd':float(x.std(ddof=1)),'standard_error':se,
            'ci95':[mean-hw,mean+hw],'half_width95':hw,'values':x.tolist()}


def signflip(values,test_n):
    # Hard errors are multiples of 1/test_n. Integer arithmetic avoids a
    # floating-point tolerance when comparing all 2^16 sign permutations.
    x=np.asarray(values,dtype=float);counts=np.rint(x*test_n).astype(np.int64)
    assert np.array_equal(counts/test_n,x)
    signs=2*((np.arange(1<<len(x),dtype=np.uint32)[:,None]>>np.arange(len(x)))&1).astype(np.int64)-1
    statistic=abs(int(counts.sum()));extreme=int(np.count_nonzero(np.abs(signs@counts)>=statistic))
    return {'p_two_sided':extreme/len(signs),'extreme_permutations':extreme,
            'total_permutations':len(signs),'absolute_sum_error_counts':statistic,
            'assumption':'Paired run differences are sign-symmetric under the null.'}


def build():
    p=read(R/'averaged_protocol.json');selection=read(R/p['selection_file'])
    out=R/'results/averaged_confirmation';allrows=read(out/'raw_results.json')
    assert len(p['confirmation_seeds'])==16
    assert len(allrows)==16*len(p['configurations'])*len(FAMILIES)
    lookup={};summaries=[];gates=[];metas={};descriptive=[]
    for cfg in p['configurations']:
        for seed in p['confirmation_seeds']:
            prefix=out/'raw'/f"{cfg['id']}_seed{seed}"
            rows=read(prefix.with_suffix('.json'));meta=read(str(prefix)+'_meta.json')
            assert len(rows)==6 and set(r['family'] for r in rows)==set(FAMILIES)
            assert rows==[r for r in allrows if r['configuration']==cfg['id'] and r['seed']==seed]
            metas[(cfg['id'],seed)]=meta
            for row in rows:
                assert row['stage']=='confirmation' and row['readout']==cfg['readout']
                key=(cfg['id'],seed,row['family']);assert key not in lookup;lookup[key]=row
                if row['family'] in p['correction_families']:
                    assert row['alpha']==selection['choices'][cfg['id']][row['family']]['alpha']
        for family in FAMILIES:
            rows=[lookup[(cfg['id'],s,family)] for s in p['confirmation_seeds']]
            entry={**cfg,'family':family}
            for metric in ['hard_error','frame_error','gain_aligned_soft_risk','average_rms_gain']:
                entry[metric]=summary([r[metric] for r in rows])
            entry['selected_alphas']=[r['alpha'] for r in rows]
            if family!='gated':assert len(set(entry['selected_alphas']))==1
            summaries.append(entry)
            if family!='averaged_sgd':
                diff=[lookup[(cfg['id'],s,family)]['hard_error']-lookup[(cfg['id'],s,'averaged_sgd')]['hard_error'] for s in p['confirmation_seeds']]
                descriptive.append({'configuration':cfg['id'],'method':family,'control':'averaged_sgd',
                                    'difference':summary(diff),'scope':'Descriptive unless included among the nine prespecified tests.'})
        gate_rows=[lookup[(cfg['id'],s,'gated')] for s in p['confirmation_seeds']]
        family_counts=Counter(r['selected_family'] for r in gate_rows)
        gates.append({'configuration':cfg['id'],'runs':16,'fallback_runs':family_counts.get('averaged_sgd',0),
                      'selected_family_counts':dict(family_counts),
                      'selected_coefficients':[{'seed':r['seed'],'family':r['selected_family'],'alpha':r['alpha']} for r in gate_rows],
                      'accepted_candidates_per_run':[sum(int(z['accepted']) for z in metas[(cfg['id'],s)]['gate_records']) for s in p['confirmation_seeds']],
                      'scope':'Delta applies separately to each trained configuration and validation sample, not jointly across this suite.'})
    tests=[]
    for cfg in p['configurations']:
        if not cfg['primary']:continue
        for method in ['avg_meancontrol','gated']:
            diff=[lookup[(cfg['id'],s,method)]['hard_error']-lookup[(cfg['id'],s,'averaged_sgd')]['hard_error'] for s in p['confirmation_seeds']]
            tests.append({'configuration':cfg['id'],'method':method,'control':'averaged_sgd',
                          'difference':summary(diff),**signflip(diff,p['test_n'])})
    diff=[lookup[('p9_clip',s,'averaged_sgd')]['hard_error']-lookup[('p9_ordinary',s,'averaged_sgd')]['hard_error'] for s in p['confirmation_seeds']]
    tests.append({'configuration':'p9_clip_vs_ordinary','method':'clipped averaged_sgd','control':'ordinary averaged_sgd',
                  'difference':summary(diff),**signflip(diff,p['test_n'])})
    assert len(tests)==9
    order=sorted(range(len(tests)),key=lambda i:tests[i]['p_two_sided']);previous=0.
    for rank,i in enumerate(order):
        adjusted=max(previous,min(1.,(len(tests)-rank)*tests[i]['p_two_sided']));previous=adjusted
        tests[i]['holm_adjusted_p']=adjusted
        mean=tests[i]['difference']['mean']
        tests[i]['outcome']='favorable' if adjusted<.05 and mean<0 else 'adverse' if adjusted<.05 and mean>0 else 'inconclusive'
    diagnostics=[]
    for cfg in p['configurations']:
        m=[metas[(cfg['id'],s)] for s in p['confirmation_seeds']]
        diagnostics.append({'configuration':cfg['id'],
                            'native_average_soft_risk':summary([z['native_average_soft_risk'] for z in m]),
                            'native_endpoint_soft_risk':summary([z['native_endpoint_soft_risk'] for z in m]),
                            'increment_variances':{name:summary([z['increment_variances'][i] for z in m]) for i,name in enumerate(['plain','online','meancontrol'])},
                            'geometry':{family:{metric:summary([z['geometry'][family][metric] for z in m]) for metric in m[0]['geometry'][family]} for family in ['plain','online','meancontrol']}})
    source_paths=['averaged_protocol.json','scripts/averaged_report.py',p['selection_file'],'results/averaged_confirmation/run_manifest.json']
    return {'study':p['study'],'status':'all_fixed_confirmation_conditions_reported','summary':summaries,'primary_tests':tests,
            'primary_outcomes':dict(Counter(z['outcome'] for z in tests)), 'gate_summary':gates,
            'descriptive_paired_differences':descriptive,'diagnostics':diagnostics,
            'inference_scope':p['primary_inference'],'secondary_scope':p['secondary_scope'],
            'source_sha256':{name:sha(R/name) for name in source_paths}}


if __name__=='__main__':
    result=build();path=R/'results/averaged_confirmation/report.json'
    if path.exists():assert read(path)==result;print('Existing averaged-SGD report reproduced exactly.')
    else:path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result['primary_outcomes']))
