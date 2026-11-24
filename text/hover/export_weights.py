"""Export only the already selected MuSiQue Majority scoring coefficients."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/"bridge"


def main():
    out={}
    for name, prev in [("eight",OLD),("one",OLD/"one_draw")]:
        path=prev/"selection_lock.json"
        lock=json.loads(path.read_text())["choices"]["majority"]
        cfg=json.loads((prev/"protocol.json").read_text())
        w0=cfg["frozen_configs"][lock["frozen"]]
        out[name]=dict(source_lock_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                       frozen=lock["frozen"],base=w0,seeds={})
        for key,cell in lock["seeds"].items():
            out[name]["seeds"][key]={arm:{"raw":run["raw_weights"],"scale":run["scale"]}
                                    for arm,run in cell["arms"].items()}
    (ROOT/"weights.json").write_text(json.dumps(out,indent=2)+"\n")


if __name__=="__main__":main()
