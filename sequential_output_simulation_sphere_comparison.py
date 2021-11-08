# -*- coding: utf-8 -*-
"""
Run a single replication of simulation with sphere settings.

PCKmeans + NPU
COPKmeans + NPU

@author: Yujia Deng
"""

import errno
import os
import sys

from sequential_output_npu import *

sys.argv = ['', 5, 100, 400, 5, 50, 10, 0]
#############
_, K, P1, P2, r, N_per_cluster, max_nc, rep = sys.argv
#############

K = int(K)
P1 = int(P1)
P2 = int(P2)
N_per_cluster = int(N_per_cluster)
r = float(r)
max_nc = int(max_nc)
rep = int(rep)

request_nc = range(10, max_nc+1, 10)

np.random.seed(rep)
save_path = './results_collection/simulation_sphere/K_%d_P1_%d_P2_%d_r_%2.1f_N_per_cluster_%d_entropy_change_lambda_200/max_nc_%d/comparison/' % (K, P1, P2, r, N_per_cluster, max_nc)
print(save_path)
print('rep=%d' % rep)

if not os.path.exists(save_path):
    try:
        os.makedirs(save_path, 0o700)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


X, y, K, num_per_class, scale = load_simulation_sphere(K, P1, P2, N_per_cluster, r, sigma=1, seed=1, random_scale=1)

print('Data generated.')

result_PCKmeans, _, _, _ = ARI_active_old(metric_learn_method='pckmeans', X=X, y=y, K=K, max_nc=max_nc, impute_method='default', weighted=False, uncertainty='random_forest',diag=True, include_H=True, lambd=0, gamma=0, rank=1, num_p=0, verbose=None, penalized=False, initial='default', request_nc=request_nc)

result_COPKmeans, _, _, _ = ARI_active_old(metric_learn_method='copkmeans', X=X, y=y, K=K, max_nc=max_nc, impute_method='default', weighted=False, uncertainty='random_forest',diag=True, include_H=True, lambd=0, gamma=0, rank=1, num_p=0, verbose=None, penalized=False, initial='default', request_nc=request_nc)


np.savez(os.path.join(save_path, ('rep%d.npz' % rep)), result_PCKmeans=result_PCKmeans,
result_COPKmeans=result_COPKmeans)

print(result_PCKmeans)
print(result_COPKmeans)

