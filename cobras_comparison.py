# -*- coding: utf-8 -*-
"""
Comparison with the COBRAS method in both simulation.

Real data comparison is added in real_data_comparison.py

@author: Yujia Deng
"""

import errno
import os
import sys

from sequential_output_npu import ARI_nc_COBRAS
from summary import output
from helper import *

###########
# Simulation 1
###########
sys.argv = ['', 5, 30, 3, 60, 300, 30]
#############
_, P1, P2, mu, N_per_cluster, max_nc, reps = sys.argv
#############

P1 = int(P1)
P2 = int(P2)
N_per_cluster = int(N_per_cluster)
mu = float(mu)
max_nc = int(max_nc)
reps = int(reps)

request_nc = range(10, max_nc+1, 10)
save_path = './results_collection/simulation_high_dim_4/P1_%d_P2_%d_mu_%2.1f_N_per_cluster_%d/max_nc_%d/comparison/COBRAS' % (P1, P2, mu, N_per_cluster, max_nc)
if not os.path.exists(save_path):
    try:
        os.makedirs(save_path, 0o700)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


results_COBRAS = np.zeros([reps, len(request_nc)])
n_query = np.zeros([reps, len(request_nc)])
for rep in range(reps):
    np.random.seed(rep)
    print(save_path)
    print('rep=%d' % rep)
    X0, y, K, num_per_class, scale = load_high_dim4(P1, P2, N=N_per_cluster, mu=mu, seed=rep)
    A0 = np.random.randn(P1+P2, P1+P2)
    A0 = A0@A0.T
    X = transform(X0, A0)

    for i, nc in enumerate(request_nc):
        results_COBRAS[rep, i], n_query[rep, i] = ARI_nc_COBRAS(X, y, nc)
        print('nc=%d, ARI_COBRAS=%2.3f, n_query=%d' % (nc, results_COBRAS[rep, i], n_query[rep, i]))

np.savez(os.path.join(save_path, 'COBRAS.npz'),
results_COBRAS=results_COBRAS, n_query=n_query)   

with open(os.path.join(save_path,'COBRAS_result.tsv'), 'w+') as file:
    file.write('number of budget')
    for nc in request_nc:
        file.write('\t' + str(nc))
    output(file, results_COBRAS, 'ARI_COBRAS')
    output(file, n_query, 'n_query')


###########
# Simulation 2
###########
sys.argv = ['', 10, 30, 3, 30, 300, 30]
#############
_, P1, P2, mu, N_per_cluster, max_nc, reps = sys.argv
#############

P1 = int(P1)
P2 = int(P2)
N_per_cluster = int(N_per_cluster)
mu = float(mu)
max_nc = int(max_nc)
reps = int(reps)

request_nc = range(10, max_nc+1, 10)
save_path = './results_collection/simulation_high_dim_4/P1_%d_P2_%d_mu_%2.1f_N_per_cluster_%d/max_nc_%d/comparison/COBRAS' % (P1, P2, mu, N_per_cluster, max_nc)
if not os.path.exists(save_path):
    try:
        os.makedirs(save_path, 0o700)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


results_COBRAS = np.zeros([reps, len(request_nc)])
n_query = np.zeros([reps, len(request_nc)])
for rep in range(reps):
    np.random.seed(rep)
    print(save_path)
    print('rep=%d' % rep)
    X0, y, K, num_per_class, scale = load_high_dim4(P1, P2, N=N_per_cluster, mu=mu, seed=rep)
    A0 = np.random.randn(P1+P2, P1+P2)
    A0 = A0@A0.T
    X = transform(X0, A0)

    for i, nc in enumerate(request_nc):
        results_COBRAS[rep, i], n_query[rep, i] = ARI_nc_COBRAS(X, y, nc)
        print('nc=%d, ARI_COBRAS=%2.3f, n_query=%d' % (nc, results_COBRAS[rep, i], n_query[rep, i]))

np.savez(os.path.join(save_path, 'COBRAS.npz'),
results_COBRAS=results_COBRAS, n_query=n_query)   

with open(os.path.join(save_path,'COBRAS_result.tsv'), 'w+') as file:
    file.write('number of budget')
    for nc in request_nc:
        file.write('\t' + str(nc))
    output(file, results_COBRAS, 'ARI_COBRAS')
    output(file, n_query, 'n_query')


###########
# Real Data
###########
max_nc, reps = 300, 30
request_nc = range(10, max_nc+1, 10)
datasets = ['breast_cancer', 'MEU-Mobile', 'urban_land_cover']
for data_name in datasets:
    X, y = load_data(data_name)
    K = len(np.unique(y))
    save_path = './real_data/%s/max_nc_%d/comparison/' % (data_name, max_nc)
    print(save_path)

    if not os.path.exists(save_path):
        try:
            os.makedirs(save_path, 0o700)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
    results_COBRAS = np.zeros([reps, len(request_nc)])
    n_query = np.zeros([reps, len(request_nc)])
    for rep in range(reps):
        np.random.seed(rep)
        print('rep=%d' % rep)
        for i, nc in enumerate(request_nc):
            results_COBRAS[rep, i], n_query[rep, i] = ARI_nc_COBRAS(X, y, nc)
            print('nc=%d, ARI_COBRAS=%2.3f, n_query=%d' % (nc, results_COBRAS[rep, i], n_query[rep, i]))

    np.savez(os.path.join(save_path, 'COBRAS.npz'),
    results_COBRAS=results_COBRAS, n_query=n_query)   

    with open(os.path.join(save_path,'COBRAS_result.tsv'), 'w+') as file:
        file.write('number of budget')
        for nc in request_nc:
            file.write('\t' + str(nc))
        output(file, results_COBRAS, 'ARI_COBRAS')
        output(file, n_query, 'n_query')

#############################
# High dimensional simulation
#############################
sys.argv = ['', 5, 100, 400, 5, 50, 300, 30]
#############
_, K, P1, P2, r, N_per_cluster, max_nc, reps = sys.argv
#############

K = int(K)
P1 = int(P1)
P2 = int(P2)
N_per_cluster = int(N_per_cluster)
r = float(r)
max_nc = int(max_nc)
reps = int(reps)

request_nc = range(10, max_nc+1, 10)
save_path = './results_collection/simulation_sphere/K_%d_P1_%d_P2_%d_r_%2.1f_N_per_cluster_%d_entropy_change_lambda_200/max_nc_%d/comparison' % (K, P1, P2, r, N_per_cluster, max_nc)
if not os.path.exists(save_path):
    try:
        os.makedirs(save_path, 0o700)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

results_COBRAS = np.zeros([reps, len(request_nc)])
n_query = np.zeros([reps, len(request_nc)])
for rep in range(reps):
    np.random.seed(rep)
    print(save_path)
    print('rep=%d' % rep)
    X, y, K, num_per_class, scale = load_simulation_sphere(K, P1, P2, N_per_cluster, r, sigma=1, seed=1, random_scale=1)
    for i, nc in enumerate(request_nc):
        results_COBRAS[rep, i], n_query[rep, i] = ARI_nc_COBRAS(X, y, nc)
        print('nc=%d, ARI_COBRAS=%2.3f, n_query=%d' % (nc, results_COBRAS[rep, i], n_query[rep, i]))

np.savez(os.path.join(save_path, 'COBRAS.npz'),
results_COBRAS=results_COBRAS, n_query=n_query)   

with open(os.path.join(save_path,'COBRAS_result.tsv'), 'w+') as file:
    file.write('number of budget')
    for nc in request_nc:
        file.write('\t' + str(nc))
    output(file, results_COBRAS, 'ARI_COBRAS')
    output(file, n_query, 'n_query')