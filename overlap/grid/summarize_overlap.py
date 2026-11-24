"""Prespecified aggregate curves and censored grid crossings, without dropping runs."""
from pathlib import Path
import csv,hashlib,json,math,statistics
import numpy as np
R=Path(__file__).resolve().parent
T5=2.570581835636314
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name,x):(R/name).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def stats(x):
    x=[float(v) for v in x];mean=statistics.mean(x);sd=statistics.stdev(x)
    half=T5*sd/math.sqrt(len(x))
    return {'mean':mean,'sd':sd,'ci95_halfwidth':half,'values':x}
def main():
    rows=json.loads((R/'confirmation/all_rows.json').read_text());p=json.loads((R/'protocol.json').read_text())
    complete=json.loads((R/'confirmation/completion.json').read_text())
    assert sha(R/'confirmation/all_rows.json')==complete['results_sha256']
    methods=['certificate','adjacent_pair','complete_sign','sign_magnitude'];curves=[];thresholds=[];contrasts=[]
    expected=len(p['seeds'])*len(p['dimensions'])*len(p['overlaps'])*len(p['depths'])*len(methods)*len(p['budgets'])
    assert len(rows)==expected and len({(r['seed'],r['d'],r['rho'],r['k'],r['method'],r['n']) for r in rows})==expected
    numeric=['clean_error','frame_error_normalized','public_values_only_error','gate_second_moment','gate_identifying_signal',
             'gate_seconds','moment_seconds','training_arithmetic_seconds','public_coefficient_seconds','shared_generation_seconds',
             'shared_teacher_seconds','process_peak_rss_mb','evaluation_seconds']
    for d in p['dimensions']:
      for rho in p['overlaps']:
       for k in p['depths']:
        for method in methods:
         group=[r for r in rows if (r['d'],r['rho'],r['k'],r['method'])==(d,rho,k,method)];cc=[]
         for n in p['budgets']:
            rr=sorted((r for r in group if r['n']==n),key=lambda r:r['seed']);assert len(rr)==6
            c={'d':d,'rho':rho,'k':k,'method':method,'n':n,**{key:stats([r[key] for r in rr]) for key in numeric}}
            cc.append(c);curves.append(c)
         crosses={}
         for metric,target in [('clean_error',p['prediction_threshold']),('frame_error_normalized',p['frame_threshold'])]:
            crossings=[]
            for seed in p['seeds']:
                found=sorted((r for r in group if r['seed']==seed and r[metric]<=target),key=lambda r:r['n'])
                hit=found[0] if found else None
                crossings.append({'seed':seed,'n_star':hit['n'] if hit else None,'right_censored':hit is None,'N_max':max(p['budgets']),
                    'training_arithmetic_seconds':hit['training_arithmetic_seconds'] if hit else None,
                    'unamortized_preprocessing_plus_training_seconds':hit['training_arithmetic_seconds']+(hit['public_coefficient_seconds'] if method!='certificate' else 0) if hit else None,
                    'gate_seconds':hit['gate_seconds'] if hit else None})
            primary=next((c for c in cc if c[metric]['mean']<=target),None)
            ordered=sorted(crossings,key=lambda x:x['n_star'] if x['n_star'] is not None else float('inf'))
            mid=ordered[2:4];median_censored=any(c['right_censored'] for c in mid)
            median_bound=sum(c['n_star'] if c['n_star'] is not None else max(p['budgets']) for c in mid)/2
            crosses[metric]={'threshold':target,'median_n_star':None if median_censored else median_bound,
                            'median_lower_bound':median_bound if median_censored else None,
                            'median_right_censored':median_censored,
                            'descriptive_mean_curve_n_star':primary['n'] if primary else None,
                            'mean_curve_right_censored':primary is None,'seeds_crossing':sum(not c['right_censored'] for c in crossings),
                            'seed_crossings':crossings,
                            'training_arithmetic_seconds_at_crossing':primary['training_arithmetic_seconds'] if primary else None,
                            'public_coefficient_seconds':cc[-1]['public_coefficient_seconds'] if method!='certificate' else stats([0]*6),
                            'preprocessing_charge_note':'Zero for certificate; full shared coefficient computation conservatively charged to each richer filter'}
         thresholds.append({'d':d,'rho':rho,'k':k,'method':method,**crosses})
        base=sorted((r for r in rows if (r['d'],r['rho'],r['k'],r['method'],r['n'])==(d,rho,k,'certificate',max(p['budgets']))),key=lambda r:r['seed'])
        for method in methods[1:]:
            comp=sorted((r for r in rows if (r['d'],r['rho'],r['k'],r['method'],r['n'])==(d,rho,k,method,max(p['budgets']))),key=lambda r:r['seed'])
            contrasts.append({'d':d,'rho':rho,'k':k,'method':method,'minus_certificate_clean_error':stats([c['clean_error']-b['clean_error'] for c,b in zip(comp,base)])})
    # Exact zero-overlap equality checks independently on saved endpoints and every budget.
    lookup={(r['seed'],r['d'],r['rho'],r['k'],r['method'],r['n']):r for r in rows}
    delta=0.
    for r in rows:
        if r['rho']==0:
            b=lookup[(r['seed'],r['d'],0.,r['k'],'certificate',r['n'])]
            delta=max(delta,abs(r['clean_error']-b['clean_error']),abs(r['frame_error_normalized']-b['frame_error_normalized']))
    assert delta==0
    write('curves_summary.json',curves);write('thresholds.json',thresholds);write('paired_contrasts.json',contrasts)
    write('audit_summary.json',{'status':'passed','expected_rows':expected,'observed_rows':len(rows),'zero_overlap_max_difference':delta,
          'source_results_sha256':sha(R/'confirmation/all_rows.json'),'completion':complete,
          'threshold_interpretation':'Primary: per-seed first grid crossing, midpoint median of six; censored central order statistics yield a strict lower bound. Mean-curve crossing is descriptive secondary.',
          'confidence_interpretation':'Pointwise 95% t intervals across six seeds, no multiplicity adjustment'})
    table=[]
    for t in thresholds:
        if t['k']==7:
            c=t['clean_error'];budget=('>'+str(c['median_lower_bound'])) if c['median_right_censored'] else str(c['median_n_star'])
            table.append([t['d'],t['rho'],t['method'],budget,c['seeds_crossing']])
    with (R/'depth7_thresholds.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['d','rho','method','median_n_star_or_lower_bound','seeds_crossing_of_6']);w.writerows(table)
    print(json.dumps({'rows':len(rows),'depth7_thresholds':table,'seconds':complete['seconds']},indent=2))
if __name__=='__main__':main()
