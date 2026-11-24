"""Convert integer sufficient statistics and an analytic infinite tail into a rational root interval."""
from fractions import Fraction as F
import json
import math
from pathlib import Path

root = Path(__file__).resolve().parent
data = json.loads((root/"counts.json").read_text())
spec = data["protocol"]
n, J, u = spec["trials"], spec["last_excursion"], spec["bernstein_u"]


def sqrt_up(x):
    scale = 10**15
    return F(math.isqrt(x.numerator*scale**2//x.denominator)+1,scale)


low, high, empirical = [], [], []
for j,(pos,neg) in enumerate(zip(data["positive"],data["negative"])):
    cap = F(1,2**(j+1))
    raw = F(pos-neg,n)
    delta = sqrt_up(2*cap*u/n)+F(4*u,3*n)
    low.append(max(F(0),raw-delta))
    high.append(min(cap,raw+delta))
    empirical.append(raw)
assert len(low)==J+1 and all(l<=h for l,h in zip(low,high))


def residual(x, coeffs, tail):
    value=0
    for c in coeffs[::-1]:
        value=(value+c)/x
    if tail:
        value += (2*x)**(-J-1)/(2*x-1)
    return x-F(1,2)-value


def grid_bounds(positive_residual):
    a,b=5001,10000
    while b-a>1:
        m=(a+b)//2
        if positive_residual(F(m,10000)):
            b=m
        else:
            a=m
    return F(a,10000),F(b,10000)


rlo=grid_bounds(lambda x:residual(x,low,False)>=0)[0]
rhi=grid_bounds(lambda x:residual(x,high,True)>=0)[1]
assert residual(rlo,low,False)<0<residual(rhi,high,True)
tlo=math.floor(-1000*math.log2(float(rhi)))
thi=math.ceil(-1000*math.log2(float(rlo)))
assert rhi**1000*2**tlo<=1<=rlo**1000*2**thi
rawroot = grid_bounds(lambda x:residual(x,empirical,False)>=0)
out=dict(
    root_interval_rational=[str(rlo),str(rhi)], root_interval=list(map(float,(rlo,rhi))),
    exponent_interval=[tlo/1000,thi/1000],
    point_root_approximation=sum(map(float,rawroot))/2,
    failure_bound=2*(J+1)*math.exp(-u),
    old_table_failure_bound=.005881,
    joint_table_failure_bound=.005881+2*(J+1)*math.exp(-u),
    tail_mass_at_upper=float((2*rhi)**(-J-1)/(2*rhi-1)),
    empirical_coefficients=list(map(float,empirical)),lower_coefficients=list(map(float,low)),upper_coefficients=list(map(float,high))
)
(root/"interval.json").write_text(json.dumps(out,indent=2)+"\n")
print({k:v for k,v in out.items() if not k.endswith("coefficients")})
