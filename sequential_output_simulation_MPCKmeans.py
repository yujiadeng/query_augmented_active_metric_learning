# -*- coding: utf-8 -*-
"""
Run a single replication of simulation.

MPCKmeans + NPU

@author: Yujia Deng
"""

from sequential_output_npu import *
import sys
import os
import errno

sys.argv = ['', 2, 4, 3, 10, 20, 0]
#############
_, P1, P2, mu, N_per_cluster, max_nc, rep = sys.argv
#############

P1 = int(P1)
P2 = int(P2)
N_per_cluster = int(N_per_cluster)
mu = float(mu)
max_nc = int(max_nc)
rep = int(rep)

request_nc = range(10, max_nc+1, 10)

np.random.seed(rep)
save_path = './results_collection/simulation_high_dim_4/P1_%d_P2_%d_mu_%2.1f_N_per_cluster_%d/max_nc_%d/comparison/MPCKmeans' % (P1, P2, mu, N_per_cluster, max_nc)
print(save_path)
print('rep=%d' % rep)

if not os.path.exists(save_path):
    try:
        os.makedirs(save_path, 0o700)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


X0, y, K, num_per_class, scale = load_high_dim4(P1, P2, N=N_per_cluster, mu=mu, seed=rep)
A0 = np.random.randn(P1+P2, P1+P2)
A0 = A0@A0.T
X = transform(X0, A0)

ARI_clustering(X, y, K)

result_MPCKmeans, _, A_hist, A_hist_penalize = ARI_active(metric_learn_method='mpckmeans', X=X, y=y, K=K, max_nc=max_nc, impute_method='default', weighted=False, uncertainty='random_forest',diag=True, include_H=True, lambd=0, gamma=0, rank=1, num_p=0, verbose=None, penalized=False, initial='default', request_nc=request_nc)

np.savez(os.path.join(save_path, ('rep%d.npz' % rep)), result_MPCKmeans=result_MPCKmeans)


print(result_MPCKmeans)
