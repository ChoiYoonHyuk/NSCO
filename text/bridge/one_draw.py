"""Freeze a one-observation sensitivity with the first bits of the same eight draws."""
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
OUTPUT=ROOT/'one_draw'

def main():
    OUTPUT.mkdir(exist_ok=True)
    for name in ['prepare.py','audit.py','data_receipt.json','raw_pairs.py']:
        shutil.copyfile(ROOT/name,OUTPUT/name)
    for p in ROOT.glob('*.private.npz'):
        link=OUTPUT/p.name
        if not link.exists(): link.symlink_to(p)
    source=(ROOT/'bridge.py').read_text()
    replacements=[
       ("'train_sign_draws_per_question':8","'train_sign_draws_per_question':1",1),
       ("and eight independent publicly observed sign draws","and one publicly observed sign draw",1),
       ('v=signs(len(order),8,rng)','v=signs(len(order),8,rng)[:,:1,:].contiguous()',1),
       ('len(order)*8','len(order)',2)
    ]
    for old,new,count in replacements:
        assert source.count(old)==count,old
        source=source.replace(old,new)
    target=OUTPUT/'bridge.py'
    if target.exists():assert target.read_text()==source
    else:target.write_text(source)
    for cmd,done in [(['bridge.py','fit'],'selection_lock.json'),(['bridge.py','evaluate'],'evaluated.json'),
                     (['audit.py'],'audit.json'),(['raw_pairs.py'],'raw_pairs.json')]:
        if not (OUTPUT/done).exists():
            subprocess.run([sys.executable,*cmd],cwd=OUTPUT,check=True)

if __name__=='__main__':main()
