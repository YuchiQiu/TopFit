import numpy as np
import sys
import os
dataset=sys.argv[1]
num_model=5
elbo=[]
for i in range(1,num_model+1):
    path = os.path.join('inference', dataset, 'elbo'+'_seed'+str(i)+'.npy')
    data = np.loadtxt(path)
    elbo.append(data)
elbo=np.asarray(elbo)
elbo=np.mean(elbo,axis=0)
np.savetxt(os.path.join('inference', dataset, 'elbo.npy'),elbo)


