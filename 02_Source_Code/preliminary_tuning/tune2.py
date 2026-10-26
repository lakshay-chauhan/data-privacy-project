import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from fl_core import *
d=load_digits(); X=d.data/16.0; y=d.target
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.3,random_state=0,stratify=y)
rff=RFFMap(64,256,gamma=1/64*2,seed=0); Ztr,Zte=rff(Xtr),rff(Xte)
ptr,pte=dirichlet_partition(ytr,yte,10,0.5,0)
R,E,B=20,1,16
r=run_federated(Ztr,ytr,Zte,yte,ptr,pte,10,rounds=R,epochs=E,batch=B,lr=10,seed=0); print('fedavg',round(r['acc'],3))
rng=np.random.RandomState(1)
print('mia fedavg',mia_auc(r['W'],Ztr,ytr,Zte,yte,rng))
for sig in [0.8,1.0,1.5,2.0,3.0]:
  for lr in [5,10]:
    r=run_federated(Ztr,ytr,Zte,yte,ptr,pte,10,rounds=R,epochs=E,batch=B,lr=lr,sigma=sig,clip=1.0,seed=0)
    eps=max(epsilon_from_rdp(*steps_per_client([p],R,E,B)[:1],sig,steps_per_client([p],R,E,B)[1]) for p in ptr)
    print('sig',sig,'lr',lr,round(r['acc'],3),'eps',round(eps,2),'mia',round(mia_auc(r['W'],Ztr,ytr,Zte,yte,rng),3))
