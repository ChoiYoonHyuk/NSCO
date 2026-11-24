"""Independent Gaussian bags, two private replicas, one integer product per excursion."""
import json
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parent
protocol = json.loads((root / "protocol.json").read_text())


def raw_bags(rng, size):
    # Coordinates are public (two candidates), private replica 1, private replica 2.
    return rng.standard_normal((6, size))


def factor(cur, nxt, plus):
    """Compute both raw half-sum/difference signs before the next bag."""
    dv = nxt[0] - nxt[1]
    public1, public2 = cur[0] * dv, cur[1] * dv
    sv1 = np.where(cur[0] > 0, 1, -1).astype(np.int8)
    sv2 = np.where(cur[1] > 0, 1, -1).astype(np.int8)
    if not plus:
        sv2 = -sv2
    both = np.ones(cur.shape[1], dtype=np.int8)
    for a,b in ((2,3),(4,5)):
        dz = nxt[a] - nxt[b]
        t1 = public1 + cur[a] * dz
        t2 = public2 + cur[b] * dz
        if np.any(t1==0) or np.any(t2==0):
            raise ValueError("Unexpected tie in a Gaussian comparison")
        term = (np.where(t1 > 0, sv1, -sv1) + np.where(t2 > 0, sv2, -sv2)) // 2
        both *= term
    return both


def main():
    n = protocol["trials"]
    batch = protocol["batch"]
    jmax = protocol["last_excursion"]
    rng = np.random.Generator(np.random.PCG64DXSM(protocol["seed"]))
    pos, neg = (np.zeros(jmax+1,dtype=np.int64) for _ in range(2))
    for block in range(0,n,batch):
        cur = raw_bags(rng,min(batch,n-block))
        product = np.ones(cur.shape[1],dtype=np.int8)
        for j in range(jmax+1):
            nxt = raw_bags(rng,product.size)
            product *= factor(cur,nxt,plus=(j==0))
            end = np.signbit(nxt[0]) != np.signbit(nxt[1])
            pos[j] += np.count_nonzero((product==1) & end)
            neg[j] += np.count_nonzero((product==-1) & end)
            live = product!=0
            if not np.any(live):
                break
            cur, product = nxt[:,live], product[live]
        if block % (32*batch)==0:
            print(f"{block+batch}/{n}",flush=True)
    result = dict(protocol=protocol,positive=pos.tolist(),negative=neg.tolist(),numpy=np.__version__)
    (root / "counts.json").write_text(json.dumps(result,indent=2)+"\n")
    print("complete",flush=True)


if __name__=="__main__":
    main()
