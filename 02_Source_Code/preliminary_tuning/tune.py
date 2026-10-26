import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from fl_core import *
d=load_digits(); X=d.data/16.0; y=d.target
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.3,random_state=0,stratify=y)
rff=RFFMap(64,256,gamma=1/64*2,seed=0); Ztr,Zte=rff(Xtr),rff(Xte)
ptr,pte=dirichlet_partition(ytr,yte,10,0.5,0)
print([len(p) for p in ptr])
for lr in [2,5,10,20]:
    r=run_federated(Ztr,ytr,Zte,yte,ptr,pte,10,rounds=30,lr=lr,seed=0)
    print('fedavg lr',lr,round(r['acc'],3))
for sig in [0.5,1.0,2.0]:
  for lr in [5,10,20]:
    r=run_federated(Ztr,ytr,Zte,yte,ptr,pte,10,rounds=30,lr=lr,sigma=sig,clip=1.0,seed=0)
    eps=max(epsilon_from_rdp(*(lambda q,s:(q,sig,s))(*steps_per_client([p],30,2,32))) for p in ptr)
    print('dp sig',sig,'lr',lr,round(r['acc'],3),'eps',round(eps,2))
