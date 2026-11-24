"""Summarize predeclared budgets and paired contrasts without dropping failed runs."""
import csv
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
def read(name):
    with (ROOT/name).open() as f:
        return list(csv.DictReader(f))

def mean_ci(x):
    mu = statistics.mean(x)
    margin = 2.776445105 * statistics.stdev(x) / math.sqrt(len(x))
    return [mu, mu-margin, mu+margin]

def summarize():
    native = read("native_results.csv")
    sign = read("sign_results.csv")
    transformer = read("transformer_results.csv")
    assert (len(native),len(sign),len(transformer)) == (85,300,100)
    result = {}
    for group in ("all","3","16"):
        pool = [r for r in native if group=="all" or r["hops"]==group]
        result["token_"+group] = {k:sum(int(r[k]) for r in pool) for k in
                                  ("sequences","tokens","errors","nonnull_char_tokens","nonnull_char_errors")}
    seeds = sorted(set(r["seed"] for r in sign))
    out = []
    for model, rows in [("router",sign),("Transformer",transformer)]:
        for g in ("first","last","parity","majority"):
            for aux in (("final","value","index") if model=="router" else ("final",)):
                group = [r for r in rows if r["readout"]==g and r.get("supervision","final")==aux]
                means = {}
                for n in [0,128,512,2048,8192]:
                    vals = [100*float(r["sign_error"]) for r in group if int(r["examples"])==n]
                    assert len(vals)==5
                    means[n] = mean_ci(vals)
                crossings = [min((int(r["examples"]) for r in group
                                  if r["seed"]==seed and float(r["sign_error"])<=.10),default=None)
                             for seed in seeds]
                out.append(dict(model=model,readout=g,supervision=aux,curve_pct=means,
                                first_crossings=crossings,success=sum(x is not None for x in crossings)))
    result["sign"]=out
    contrasts={}
    for g in ("first","last","parity","majority"):
        byarm = {(r["seed"],r["supervision"]):100*float(r["sign_error"]) for r in sign
                 if r["readout"]==g and r["examples"]=="2048"}
        contrasts[g] = {arm:mean_ci([byarm[seed,arm]-byarm[seed,"final"] for seed in seeds])
                        for arm in ("value","index")}
    result["paired_aux_minus_final_pp_at_2048"]=contrasts
    result["transformer_parameters"]=int(transformer[0]["parameters"])
    audit=json.loads((ROOT/"independent_audit.json").read_text())
    assert all(x["matched"] for group in audit.values() for x in group)
    (ROOT/"summary.json").write_text(json.dumps(result,indent=2)+"\n")
    tab=[]
    for row in out:
        a={"final":"Final","value":r"$+C_1$","index":r"$+I_1$"}[row["supervision"]]
        mu=[row["curve_pct"][n][0] for n in (512,2048,8192)]
        tab.append(f'{row["model"].capitalize()} & {row["readout"].capitalize()} & {a} & '
                   + " & ".join(f"{v:.2f}" for v in mu)+f' & {row["success"]}/5'+r'\\')
    (ROOT/"hop_table.tex").write_text("\n".join(tab)+"\n")
    for row in out:
        print(row["model"],row["readout"],row["supervision"],round(row["curve_pct"][8192][0],2),row["first_crossings"])
    print("paired",contrasts)

if __name__=="__main__":
    summarize()
