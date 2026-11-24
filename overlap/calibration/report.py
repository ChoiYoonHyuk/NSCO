"""Generate paired comparisons and the reviewer-facing LaTeX fragment from frozen records."""
from pathlib import Path
import csv,json,math,statistics,hashlib
ROOT=Path(__file__).resolve().parent;O=ROOT/'production'
def read(f):return list(csv.DictReader(f.open()))
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
def save(f,x):f.write_text(json.dumps(x,indent=2)+'\n')
def dump(f,rows):
    with f.open('w',newline='') as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def ci(v):
    b=statistics.mean(v);err=2.5705818366*statistics.stdev(v)/math.sqrt(len(v))
    return [b,b-err,b+err]
def main():
    p=json.loads((ROOT/'protocol.json').read_text());audit=json.loads((O/'completion.json').read_text())
    for f,h in audit['output_sha256'].items():assert sha(O/f)==h
    rows=sum((read(O/f'outcomes_{s}.csv') for s in p['seeds']),[])
    ix={(int(z['seed']),float(z['true_rho']),z['arm'],int(z['n'])):z for z in rows}
    assert len(ix)==len(rows)==audit['rows']
    cap=max(p['budgets']);summary=[];cross=[];contrasts=[];snap={}
    metrics=['clean_error','normalized_frame_error','full_path_accuracy','chosen_rho','norm_abs_error','validation_seconds']
    for r in p['overlaps']:
        for a in p['arms']:
            first={}
            for ep,field,cutoff in [('hard','clean_error',p['hard_threshold']),('frame','normalized_frame_error',p['frame_threshold'])]:
                obs=[]
                for s in p['seeds']:
                    n=next((n for n in p['budgets'] if float(ix[s,r,a,n][field])<=cutoff),None)
                    obs.append(n);cross.append(dict(seed=s,rho=r,arm=a,metric=ep,cutoff=cutoff,n_first=n,right_censored=n is None,cap=cap))
                successes=sum(n is not None for n in obs)
                first[ep+'_crossings']=successes
                first[ep+'_median_or_lower_bound']=statistics.median([n or cap for n in obs])
                first[ep+'_is_lower_bound']=successes<=3
            for n in p['budgets']:
                row=dict(rho=r,arm=a,n=n)
                for field in metrics:
                    values=[float(ix[s,r,a,n][field]) for s in p['seeds']]
                    mu,lo,hi=ci(values)
                    row[field]=mu;row[field+'_ci_lo']=lo;row[field+'_ci_hi']=hi
                rv=[float(ix[s,r,a,n]['angular_error_degrees']) for s in p['seeds'] if ix[s,r,a,n]['angular_error_degrees']!='']
                row['angular_error_degrees_nonzero']=statistics.mean(rv) if rv else None
                row['nonzero_selected']=len(rv)
                row['selected_norms']=[float(ix[s,r,a,n]['chosen_rho']) for s in p['seeds']]
                summary.append(row)
                if n==cap:snap[r,a]={**row,**first}
            if a=='oracle':continue
            for base in ['oracle','short_given_direction']:
                if base==a:continue
                for m in metrics[:3]:
                    v=[float(ix[s,r,a,cap][m])-float(ix[s,r,base,cap][m]) for s in p['seeds']]
                    mu,lo,hi=ci(v);contrasts.append(dict(rho=r,n=cap,arm=a,comparison=base,metric=m,mean=mu,ci95_lo=lo,ci95_hi=hi,seed_differences=v))
    dump(ROOT/'crossings.csv',cross);dump(ROOT/'paired.csv',contrasts);dump(ROOT/'curves.csv',summary)
    save(ROOT/'summary.json',dict(endpoint_cells=list(snap.values()),paired_endpoint_contrasts=contrasts,per_seed_crossings=cross))
    print('Wrote summary.json, paired.csv, curves.csv and crossings.csv.')

if __name__ == '__main__':
    main()
