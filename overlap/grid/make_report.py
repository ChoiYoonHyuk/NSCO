from pathlib import Path
import json,statistics,sys,math
R=Path(__file__).resolve().parent
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
T=json.loads((R/'thresholds.json').read_text());C=json.loads((R/'curves_summary.json').read_text());P=json.loads((R/'protocol.json').read_text())
methods=['certificate','adjacent_pair','complete_sign','sign_magnitude']
names=['Certificate','Adjacent pair','Complete sign','Sign + magnitude']
colors=['#6b7280','#b45309','#2563eb','#0f766e']
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
def plotting(rhos,name,size):
    fig,axes=plt.subplots(2,len(rhos),figsize=size,squeeze=False,sharey=True)
    for row,d in enumerate([2,4]):
      for col,rho in enumerate(rhos):
        ax=axes[row,col]
        for j,m in enumerate(methods):
            rs=[next(x for x in T if (x['d'],x['rho'],x['k'],x['method'])==(d,rho,k,m)) for k in P['depths']]
            xs=np.array(P['depths'])+(j-1.5)*.06;ys=[]
            for x,r in zip(xs,rs):
                t=r['clean_error'];y=math.log2(t['median_n_star']) if not t['median_right_censored'] else 20.3
                ys.append(y);ax.scatter(x,y,color=colors[j],marker='^' if t['median_right_censored'] else 'o',s=34,zorder=3)
            ax.plot(xs,ys,color=colors[j],lw=1.3,label=names[j])
        ax.axhline(20,color='#b6bbc3',lw=.7,ls=':');ax.set_title(r'$d=%d,\ \rho=%g$'%(d,rho))
        ax.set_xticks(P['depths']);ax.set_ylim(10.5,20.8);ax.set_yticks([12,14,16,18,20]);ax.grid(axis='y',alpha=.18)
        if row==1:ax.set_xlabel('Depth k')
        if col==0:ax.set_ylabel(r'$\log_2$ median labels to error $\leq 1/8$')
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=4,frameon=False,bbox_to_anchor=(.5,.01))
    fig.text(.5,.065,'Upward triangles: median right-censored beyond the maximum budget $2^{20}$.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.10,1,1]);fig.savefig(R/(name+'.png'),dpi=220);fig.savefig(R/(name+'.pdf'));plt.close(fig)
plotting([0.,.5,1.],'overlap_target_cost_grid',(11.3,7.1))
report=[]
for d in [2,4]:
  for k in [3,5,7]:
    for m in methods:
      t=next(t for t in T if (t['d'],t['rho'],t['k'],t['method'])==(d,1.,k,m))['clean_error']
      c=next(c for c in C if (c['d'],c['rho'],c['k'],c['method'],c['n'])==(d,1.,k,m,2**20))
      hits=[r for r in t['seed_crossings'] if not r['right_censored']]
      report.append({'d':d,'rho':1.,'k':k,'method':m,'median_n_star':t['median_n_star'],'median_lower_bound':t['median_lower_bound'],
          'crossing_count':len(hits),'seed_n_star':[r['n_star'] for r in t['seed_crossings']],
          'mean_arithmetic_seconds_at_per_seed_crossing':statistics.mean(r['training_arithmetic_seconds'] for r in hits) if hits else None,
          'mean_preprocessing_plus_arithmetic_seconds_at_crossing':statistics.mean(r['unamortized_preprocessing_plus_training_seconds'] for r in hits) if hits else None,
          'endpoint_error':c['clean_error'],'public_only_error':c['public_values_only_error'],'endpoint_frame':c['frame_error_normalized'],
          'preprocessing_seconds':t['public_coefficient_seconds']['mean'],
          'endpoint_gate_microseconds_per_example':1e6*c['gate_seconds']['mean']/2**20})
(R/'headline_numbers.json').write_text(json.dumps(report,indent=2)+'\n')
for d in [2,4]:
    for m in methods:
        row=next(r for r in report if (r['d'],r['k'],r['method'])==(d,7,m))
        print(json.dumps(row))
