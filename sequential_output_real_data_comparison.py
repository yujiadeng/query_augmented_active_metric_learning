# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 17:21:07 2020

Run a single replication of simulation.

PCKmeans + NPU
COPkmeans + NPU

@author: Yujia Deng
"""

from sequential_output_npu import ARI_active
import numpy as np
import sys
import os
import pandas as pd
from sklearn.decomposition import PCA
import errno
from helper import load_data

sys.argv = '', 'breast_cancer', 300, 0
#############
_, data_name, max_nc, rep = sys.argv
#############



max_nc = int(max_nc)
rep = int(rep)

request_nc = range(10, max_nc+1, 10)

np.random.seed(rep)
save_path = './real_data/%s/max_nc_%d/comparison/' % (data_name, max_nc)
print(save_path)
print('rep=%d' % rep)

if not os.path.exists(save_path):
    try:
        os.makedirs(save_path, 0o700)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

X, y = load_data(data_name)
K = len(np.unique(y))

result_PCKmeans, _, _, _ = ARI_active(metric_learn_method='pckmeans', X=X, y=y, K=K, max_nc=max_nc, impute_method='default', weighted=False, uncertainty='random_forest',diag=True, include_H=True, lambd=0, gamma=0, rank=1, num_p=0, verbose=None, penalized=False, initial='default', request_nc=request_nc)

result_COPKmeans, _, _, _ = ARI_active(metric_learn_method='copkmeans', X=X, y=y, K=K, max_nc=max_nc, impute_method='default', weighted=False, uncertainty='random_forest',diag=True, include_H=True, lambd=0, gamma=0, rank=1, num_p=0, verbose=None, penalized=False, initial='default', request_nc=request_nc)


np.savez(os.path.join(save_path, ('rep%d.npz' % rep)), result_PCKmeans=result_PCKmeans, result_COPKmeans=result_COPKmeans)

print(result_PCKmeans)
print(result_COPKmeans)
