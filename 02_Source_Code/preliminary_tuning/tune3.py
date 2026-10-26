import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from fl_core import *
d=load_digits(); X=d.data/16.0; y=d.target
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.7,random_state=0,stratify=y)
rff=RFFMap(64,1024,gamma=1/64*2,seed=0); Ztr,Zte=rff(Xtr),rff(Xte)
ptr,pte=dirichlet_partition(ytr,yte,10,0.5,0)
print(len(ytr),[len(p) for p in ptr])
rng=np.random.RandomState(1)
for R,E in [(40,3),(60,5)]:
    r=run_federated(Ztr,ytr,Zte,yte,ptr,pte,10,rounds=R,epochs=E,batch=16,lr=20,seed=0)
    print(R,E,'fedavg acc',round(r['acc'],3),'train acc',round(accuracy(r['W'],Ztr,ytr),3),'mia',round(mia_auc(r['W'],Ztr,ytr,Zte,yte,rng),3))
