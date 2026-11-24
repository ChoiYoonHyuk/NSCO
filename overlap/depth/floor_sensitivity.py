"""Replay the six archived first batches once; add a prespecified lower budget grid.

Settings fixed before looking at any within-batch crossing:
all original overlaps and depths, D=16, archived sign-and-magnitude learner,
original frame-error endpoint .35, powers of two from 32 to 2048.
The original 4096 checkpoint is an independent replay checksum for each cell.
"""
import csv
import json
import math
from pathlib import Path
import sys

root = Path(__file__).resolve().parent
archive = Path(sys.argv[1])
sys.path.insert(0,str(archive))
from depth import basis, draw, fit, gate, latent_path, np, rng, rowspec, sign, target


def interval(values):
    mu=float(np.mean(values))
    width=2.57058183563631*float(np.std(values,ddof=1))/math.sqrt(6)
    return dict(mean=mu,ci95=[mu-width,mu+width],values=values)


def run():
    spec=json.loads((archive/"protocol.json").read_text())
    raw_coeff=json.loads((archive/"production/coefficients.json").read_text())
    overlap_grid=sorted({s[4] for s in rowspec(spec)})
    coeff=dict(zip(overlap_grid,raw_coeff))
    D=16
    additional=[2**i for i in range(5,12)]
    assert spec["batch_size"]==4096 and spec["budgets"][0]==4096
    verified=0
    maxdiff=0
    checks=[]
    allcross={}
    for seed in spec["seeds"]:
        base=basis(seed,D)
        train=rng(seed,302)
        q,g,z,nu=draw(train,4096,max(spec["depths"]),max(spec["ambient"]),False)
        noise=np.where(train.random(len(q))<(1+spec["lambda"])/2,1,-1)
        zz=(z[:,0,0]-z[:,0,1])/math.sqrt(2)
        left=(g[:,0,0]-g[:,0,1])/math.sqrt(2)
        xx=np.concatenate([zz,nu],axis=1)
        R=(xx[:,:D-1]@base.T)*left[:,None]
        b=sign(g[:,1:]).mean(2)
        with (archive/f"production/{seed}.csv").open() as f:
            archived={(float(row["rho"]),int(row["k"]),int(row["n"])):float(row["frame_error_normalized"])
                      for row in csv.DictReader(f) if int(row["D"])==D and row["method"]=="sign_magnitude" and row["setting"]=="exact"}
        for rho in spec["overlaps"]:
            truth=latent_path(q,g,z,rho)[1]
            trueP=target(base[:,:2],rho)
            for k in spec["depths"]:
                scale=gate(b,g,k,"sign_magnitude",coeff[rho])
                yq=(truth[:,:k].prod(1)*noise*scale)[:,None]*q
                errs={n:archived[rho,k,n] for n in spec["budgets"]}
                for n in additional+[4096]:
                    P=fit(R[:n].T@yq[:n]/n,rho,0)
                    err=float(np.linalg.norm(P-trueP)/math.sqrt(2))
                    if n==4096:
                        diff=abs(err-errs[n]); maxdiff=max(maxdiff,diff)
                        assert diff<1e-10,(rho,k,seed,err,errs[n])
                        verified+=1
                    else:
                        errs[n]=err
                    checks.append([seed,rho,k,n,err])
                first=lambda ns:next((n for n in ns if errs[n]<=spec["frame_threshold"]),None)
                allcross[seed,rho,k]=(first(spec["budgets"]),first(additional+spec["budgets"]))
    slopes={}
    for rho in spec["overlaps"]:
        for ix,name in enumerate(["old","extended"]):
            lo=[allcross[s,rho,3][ix] for s in spec["seeds"]]
            hi=[allcross[s,rho,9][ix] for s in spec["seeds"]]
            vals=[math.log2(y/x)/6 for x,y in zip(lo,hi)]
            slopes[str(rho)+":"+name]=dict(shallow=lo,deep=hi,**interval(vals))
    for name in ["old","extended"]:
        slopes[name+"_full_minus_zero"]=interval([x-y for x,y in zip(slopes["1.0:"+name]["values"],
                                                                     slopes["0.0:"+name]["values"])])
    out=dict(confirmed_original_values=verified,maximum_absolute_checksum_difference=maxdiff,
             lower_grid=additional,method="sign_magnitude",D=D,threshold=spec["frame_threshold"],slopes=slopes)
    (root/"floor_sensitivity.json").write_text(json.dumps(out,indent=2)+"\n")
    with (root/"prefix_frame_errors.csv").open("w",newline="") as f:
        w=csv.writer(f);w.writerow(["seed","rho","k","n","frame_error"]);w.writerows(checks)
    print(json.dumps(out))


if __name__=="__main__":
    run()
