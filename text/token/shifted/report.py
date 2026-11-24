"""Reproduce paired endpoint contrasts and TeX from saved seed-level measurements."""
import csv,json,math,statistics,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def read(name):return list(csv.DictReader((ROOT/name).open()))
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
def ci(v):
    x=statistics.mean(v); d=2.776445105*statistics.stdev(v)/math.sqrt(5)
    return [x,x-d,x+d]
def raw_ci(row):
    v,lo,hi=row
    return f'{v:.2f} (95\\% paired $t$ interval $[{lo:.2f},{hi:.2f}]$)'
def main():
    p=json.loads((ROOT/'protocol.json').read_text());audit=json.loads((ROOT/'audit.json').read_text())
    seed=p['seeds'];rows=read('results.csv')
    assert len(rows)==200 and sha(ROOT/'results.csv')==audit['results_sha256']
    for f,h in audit['source_sha256'].items():assert sha(ROOT/f)==h
    for a in audit['frames']:assert sha(ROOT/f'fits_{a["seed"]}.npz')==a['frames_sha256']
    idx={(int(v['seed']),v['alignment'],v['readout'],int(v['n'])):v for v in rows}
    assert len(idx)==200
    metrics=['hard_error','soft_error','full_path_accuracy','hop1','hop2','hop3']
    summary={};pairs=[];cross=[];tab=[]
    for g in p['readouts']:
        summary[g]={}
        for a in p['alignment']:
            cells={}
            for n in p['budgets']:
                cells[str(n)]={m:ci([100*float(idx[s,a,g,n][m]) for s in seed]) for m in metrics}
            for s in seed:
                first=next((n for n in p['budgets'] if float(idx[s,a,g,n]['hard_error'])<=.1),None)
                cross.append(dict(seed=s,alignment=a,readout=g,n_first=first,censored_at=None if first is not None else 8192))
            summary[g][a]=cells
        for n in p['primary_budgets']:
            for m in metrics:
                d=[100*(float(idx[s,'successor',g,n][m])-float(idx[s,'same_position',g,n][m])) for s in seed]
                mean,lo,hi=ci(d);pairs.append(dict(readout=g,n=n,metric=m,mean_difference_pp=mean,ci95_lo=lo,ci95_hi=hi,seed_differences=d))
        for n in p['primary_budgets']:
            row=[g.capitalize(),str(n)]
            for a in p['alignment']:
                row.extend(f'{summary[g][a][str(n)][v][0]:.2f}' for v in ['hard_error','full_path_accuracy'])
            tab.append(' & '.join(row)+r'\\')
    file=ROOT/'contrasts.csv'
    with file.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(pairs[0]));w.writeheader();w.writerows(pairs)
    ref_s=read('../sign_results.csv');ref_t=read('../transformer_results.csv')
    anchor={}
    for g in p['readouts']:
        anchor[g]={}
        for n in p['primary_budgets']:
            base=[100*float(v['sign_error']) for v in ref_s if v['examples']==str(n) and v['readout']==g and v['supervision']=='final']
            trans=[100*float(v['sign_error']) for v in ref_t if v['examples']==str(n) and v['readout']==g]
            assert len(base)==len(trans)==5
            anchor[g][str(n)]={'table_router':ci(base),'causal_transformer':ci(trans)}
    result={'units':'percentage points; within-cell means and two-sided seed-level 95% t intervals',
            'cells':summary,'paired_successor_minus_same_position':pairs,'reference_final_only_error':anchor,'target_crossings':cross}
    (ROOT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    def endpoint(g,align,n,metric):return summary[g][align][str(n)][metric][0]
    def contrast(g,n,m):
        r=next(x for x in pairs if (x['readout'],x['n'],x['metric'])==(g,n,m))
        return raw_ci([r['mean_difference_pp'],r['ci95_lo'],r['ci95_hi']])
    paras=[r'\paragraph{Learned attention with and without successor-aligned keys.}\label{app:shifted-hop-attention}',
        r'We trained two 33-parameter attention models on the same signed three-hop examples. Both learn query and key embeddings $Q,K\in\mathbb{R}^{4\times4}$ and a common recency coefficient, share their transition across three hops, and marginalize the known readout over the soft paths during training. An eligible candidate position $j<t$ has score $Q_{x_t}^{\top} K_{x_{j-\delta}}/2-\operatorname{softplus}(a)\log(1+t-j)$. Setting $\delta=1$ aligns the key with the character preceding the candidate value. Setting $\delta=0$ gives the same-position control. The two models share the same causal candidate mask, initial embeddings, optimizer, label stream, and fixed evaluation queries. Public signs enter only the readout. Training uses final labels, Adam at $.035$, and one pass. Deployment follows the highest-scoring candidate at each hop.',
        r'\begin{table}[t]\centering\small\caption{Signed hop-3 hard answer error (Err.) and complete path accuracy (Path), in percent, averaged over five paired seeds. Each model uses 33 trainable parameters.}\label{tab:shifted-hop-attention}\begin{tabular}{lrrrrr}\toprule'
        +'\n'+r' & & \multicolumn{2}{c}{Successor key} & \multicolumn{2}{c}{Same-position key}\\'
        +'\n'+r'Readout & Labels & Err. & Path & Err. & Path\\\midrule'
        +'\n'+'\n'.join(tab)+'\n'+r'\bottomrule\end{tabular}\end{table}',
        r'At 8,192 labels the parity answer error was '
        +f"{endpoint('parity','successor',8192,'hard_error'):.2f}\\% for successor-aligned keys and {endpoint('parity','same_position',8192,'hard_error'):.2f}\\% for same-position keys. "
        +r'The paired change in error, measured in percentage points, was '+contrast('parity',8192,'hard_error')+'. '
        +r'The paired change in full-path accuracy was '+contrast('parity',8192,'full_path_accuracy')+'. '
        +r'At 2,048 labels alignment alone left parity error at '
        +f"{endpoint('parity','successor',2048,'hard_error'):.2f}\\% and last-value error at {endpoint('last','successor',2048,'hard_error'):.2f}\\%. "
        +r'The final-only 17-parameter match-table router and the 41,537-parameter causal Transformer reached parity errors of '
        +f"{anchor['parity']['8192']['table_router'][0]:.2f}\\% and {anchor['parity']['8192']['causal_transformer'][0]:.2f}\\%, respectively. "
        +r'The paired attention comparison isolates the candidate-key alignment within this family. The Transformer also differs in its readout and transition parameterization.']
    fp=ROOT/'fresh_receipt.json'
    if fp.is_file():
        fresh=json.loads(fp.read_text());assert fresh['per_seed_csv_sha256']==sha(ROOT/'fresh_counts.csv')
        assert fresh['frozen_protocol_sha256']==sha(ROOT/'holdout_protocol.json')
        assert fresh['source_sha256']==sha(ROOT/'fresh_check.py')
        d={(z['alignment'],z['readout']):z for z in fresh['aggregates']}
        assert all(z['examples']==10240 for z in d.values())
        def ints(a,field):return ', '.join(f"{d[a,g][field]:,}" for g in p['readouts'])
        aa=d['successor','parity'];bb=d['same_position','parity']
        paras.append(r'\paragraph{Fresh evaluation of the frozen models.} '
            r'We then fixed all 40 fitted models and evaluated 2,048 new sequences per seed (10,240 total) from the released generator with fresh queries and signs. '
            +r'For First, Last, Parity and Majority, respectively, answer-error counts out of 10,240 were '
            +ints('successor','hard_answer_errors')+' with successor-aligned keys and '
            +ints('same_position','hard_answer_errors')+' with same-position keys. '
            +f"On parity the models recovered all three positions on {aa['whole_path_correct']:,}/10,240 and {bb['whole_path_correct']:,}/10,240 sequences, respectively. "
            +r'This second evaluation used the original parameter files without further training.')
    (ROOT/'manuscript_appendix.tex').write_text('\n\n'.join(paras)+'\n',encoding='utf-8')
    print(paras[-1])
    print('successes',[(g,a,sum(v['n_first'] is not None for v in cross if v['readout']==g and v['alignment']==a)) for g in p['readouts'] for a in p['alignment']])
if __name__=='__main__':main()

