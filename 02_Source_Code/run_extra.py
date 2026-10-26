"""run_extra.py - robustness of centralised and local-only baselines under 20% label-flipped clients (for trust score)."""
import json, os, numpy as np
os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"), exist_ok=True)
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from fl_core import *
R,E,B,LR=20,1,16,10.0
out={"central_clean":[], "central_20":[], "local_clean":[], "local_20":[]}
for s in [0,1,2]:
    d=load_digits(); X,y=d.data/16.0,d.target
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.3,random_state=s,stratify=y)
    rff=RFFMap(64,256,gamma=2.0/64,seed=s); Ztr,Zte=rff(Xtr),rff(Xte)
    ptr,pte=dirichlet_partition(ytr,yte,10,1.0,s); rng=np.random.RandomState(50+s)
    for tag,frac in (("clean",0.0),("20",0.2)):
        mal=set(range(int(round(frac*10)))); yf=ytr.copy()
        for k in mal: yf[ptr[k]]=(ytr[ptr[k]]+1)%10
        Wc,_,_=local_train(np.zeros((Ztr.shape[1],10)),Ztr,yf,10,R*E,B,LR,rng)
        out["central_"+tag].append(accuracy(Wc,Zte,yte))
        accs=[]
        for k in range(10):
            Wk,_,_=local_train(np.zeros((Ztr.shape[1],10)),Ztr[ptr[k]],yf[ptr[k]],10,R*E,B,LR,rng)
            accs.append(accuracy(Wk,Zte,yte))
        out["local_"+tag].append(float(np.mean(accs)))
out={k:[float(np.mean(v)),float(np.std(v))] for k,v in out.items()}
json.dump(out,open("../results/extra_robustness.json","w"),indent=1); print(out)
