# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 17:21:07 2020

Run a single replication of simulation.
Output the ARI of the proposed method w/ and wo penalty and NPU + MPCKmeans, and COBRA

@author: Yujia Deng
"""

from sequential_output_npu import ARI_active, ARI_nc_COBRA
import numpy as np
import sys
import os
import pandas as pd
from sklearn.decomposition import PCA
import errno
from sklearn.datasets import load_wine, load_digits, load_iris, load_breast_cancer, fetch_olivetti_faces, fetch_covtype
from helper import load_data

sys.argv = '', 'breast_cancer', 300, 0
#############
_, data_name, max_nc, rep = sys.argv
#############


max_nc = int(max_nc)
rep = int(rep)

request_nc = range(10, max_nc+1, 10)

np.random.seed(rep)
save_path = './real_data/%s_entropy_change/max_nc_%d/' % (data_name, max_nc)
print(save_path)
print('rep=%d' % rep)

if not os.path.exists(save_path):
    try:
        os.makedirs(save_path, 0o700)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

X, y = load_data(data_name)
X = X - np.mean(X, 0)
K = len(np.unique(y))


result_MPCKmeans, _, _, _ = ARI_active(metric_learn_method='mpckmeans', X=X, y=y, K=K, max_nc=max_nc, impute_method='default', weighted=True, uncertainty='random_forest',diag=True, include_H=True, lambd=0, gamma=0, rank=1, num_p=0, verbose=None, penalized=False, initial='default', request_nc=request_nc)

result_proposed_no_penalty, result_proposed_penalty, A_hist, A_hist_penalize = ARI_active(metric_learn_method='proposed', X=X, y=y, K=K, max_nc=max_nc, impute_method='default', weighted=True, uncertainty='random_forest',diag=True, true_H=False, include_H=True, lambd=1000, gamma=100, rank=1, num_p=int(10), verbose=None, penalized=True, initial='default', request_nc=request_nc)

###########
# For COBRA
###########
n_super_instances = range(10, 300, 20)
result_COBRA = dict()
for nsi in n_super_instances:
    ARI, n_query = ARI_nc_COBRA(X, y, K, nsi)
    result_COBRA[nsi] = (n_query, ARI)

np.savez(os.path.join(save_path, ('rep%d.npz' % rep)), result_MPCKmeans=result_MPCKmeans, result_proposed_no_penalty=result_proposed_no_penalty, result_proposed_penalty=result_proposed_penalty, result_COBRA=result_COBRA, A_hist=A_hist, A_hist_penalize=A_hist_penalize)


print(result_MPCKmeans)
print(result_proposed_no_penalty)
print(result_proposed_penalty)
print(result_COBRA)
