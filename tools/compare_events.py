"""Match two events.csv files one-to-one (same direction, |dt| <= tolerance) and list the rest."""
import argparse,csv,json
from pathlib import Path
from scipy.optimize import linear_sum_assignment
import numpy as np

def load(p):
    with open(p) as f:return list(csv.DictReader(f))

def match(a,b,tol):
    if not a or not b:return [],list(range(len(a))),list(range(len(b)))
    cost=np.full((len(a),len(b)),1e6)
    for i,x in enumerate(a):
        for j,y in enumerate(b):
            dt=abs(float(x['seconds'])-float(y['seconds']))
            if x['direction']==y['direction'] and dt<=tol:cost[i,j]=dt
    r,c=linear_sum_assignment(cost);pairs=[(i,j) for i,j in zip(r,c) if cost[i,j]<1e6]
    mi={i for i,_ in pairs};mj={j for _,j in pairs}
    return pairs,[i for i in range(len(a)) if i not in mi],[j for j in range(len(b)) if j not in mj]

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('a');p.add_argument('b');p.add_argument('--tol',type=float,default=1.5);p.add_argument('--json',type=Path)
    x=p.parse_args();a,b=load(x.a),load(x.b);pairs,ua,ub=match(a,b,x.tol)
    print(f'matched {len(pairs)}  only_a {len(ua)}  only_b {len(ub)}')
    for i in ua:print('A',a[i]['direction'],a[i]['seconds'],a[i]['x'],a[i]['y'])
    for j in ub:print('B',b[j]['direction'],b[j]['seconds'],b[j]['x'],b[j]['y'])
