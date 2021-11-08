# -*- coding: utf-8 -*-
"""
Compare the performance of active_query, random_query and MPCKmeans parallely

@author: Yujia Deng
"""
from Step1_Impute import infer_membership_from_label
from Step2_Metric_learn import metric_learn_A
from Step3_Query import inferred_query
from helper import *
# from combine import active_metric_learn
from active_semi_clustering.exceptions import EmptyClustersException
import time
from active_MPCKmeans import *
from active_semi_clustering.active.pairwise_constraints import ExampleOracle, MinMax, NPU, ExploreConsolidate
from cobra.src.cobra import *
import random

def ARI_nc_COBRA(X, y, K, n_super_instance):
    clusterer = COBRA(n_super_instance)
    clusterings, _, ml, cl = clusterer.cluster(X, y, range(len(y)))
    return adjusted_rand_score(y, clusterings[-1]), len(ml) + len(cl)

def repeat_COBRA(n_super_instance, rep, P1, P2):
    print("rep=%d, n_super_instance=%d" % (rep, n_super_instance))
    X0, y, K, num_per_class, scale = load_high_dim3(P1,P2,10,mu=5, seed=rep, random_scale=False)
    X = X0
    N, p = X.shape
    ARI, n_query = ARI_nc_COBRA(X, y, K, n_super_instance)
    print('n_query=%d, ARI_COBRA=%2.3f' %(n_query, ARI))
    return (n_query, ARI)


def ARI_semi_active(X, y, K, nc, semi, active):
    oracle = ExampleOracle(y, max_queries_cnt=nc)
    if active.lower() == 'minmax':
        active_learner = MinMax(n_clusters=K)
        if semi.lower() == 'pckmeans':
            clusterer = PCKMeans(n_clusters = K)
        elif semi.lower() == 'mpckmeans':
            clusterer = MPCKMeans(n_clusters = K)
            
    elif active.lower() == 'npu':      
        if semi.lower() == 'pckmeans':
            clusterer = PCKMeans(n_clusters = K)
        elif semi.lower() == 'mpckmeans':
            clusterer = MPCKMeans(n_clusters = K)
        elif semi.lower() == 'mpckmeansmf':
            clusterer = MPCKMeansMF(n_clusters = K)
        active_learner = NPU(clusterer=clusterer)
        
        
    active_learner.fit(X, oracle)
    pairwise_constraints = active_learner.pairwise_constraints_
    clusterer.fit(X, ml=pairwise_constraints[0], cl=pairwise_constraints[1])
    return adjusted_rand_score(y, clusterer.labels_)  

def ARI_semi_active_with_constraints(X, y, K, nc, semi, active):
    oracle = ExampleOracle(y, max_queries_cnt=nc)
    if active.lower() == 'minmax':
        active_learner = MinMax(n_clusters=K)
        if semi.lower() == 'pckmeans':
            clusterer = PCKMeans(n_clusters = K)
        elif semi.lower() == 'mpckmeans':
            clusterer = MPCKMeans(n_clusters = K)
            
    elif active.lower() == 'npu':      
        if semi.lower() == 'pckmeans':
            clusterer = PCKMeans(n_clusters = K)
        elif semi.lower() == 'mpckmeans':
            clusterer = MPCKMeans(n_clusters = K)
        elif semi.lower() == 'mpckmeansmf':
            clusterer = MPCKMeansMF(n_clusters = K)
        active_learner = NPU(clusterer=clusterer)
        active_learner.get_true_label(y)
        
    active_learner.fit(X, oracle)
    pairwise_constraints = active_learner.pairwise_constraints_
    clusterer.fit(X, ml=pairwise_constraints[0], cl=pairwise_constraints[1])
    return adjusted_rand_score(y, clusterer.labels_), pairwise_constraints, active_learner.sequential_constraints


def comparison_high_dim(nc, rep, P1, P2, nc_per_step, true_H=False):
    print('rep=%d, number of constraints= %d' % (rep, nc))
    X0, y, K, num_per_class, scale = load_high_dim3(P1,P2,10,mu=5, seed=rep, random_scale=False)
    X = X0
    N, p = X.shape  
#   MinMax + PCKmeans
    while True:
        try:
            t0 = time.time()
            ARI_PCKmeans_MinMax = ARI_semi_active(X, y, K, nc, 'PCKmeans', 'MinMax')
            t1 = time.time()
            print("PCKMeans+MinMax costs %2.3fs" % (t1-t0))
            break
        except EmptyClustersException:
            print('Clustering Error')                
    while True:
        try:
            t0 = time.time()
            ARI_PCKmeans_NPU = ARI_semi_active(X, y, K, nc, 'PCKmeans', 'NPU')
            t1 = time.time()
            print("PCKMeans+NPU costs %2.3fs" % (t1-t0))
            break
        except EmptyClustersException:
            print('Clustering Error')
    while True:
        try:
            t0 = time.time()
            ARI_MPCKmeans_MinMax = ARI_semi_active(X, y, K, nc, 'MPCKmeans', 'minmax')
            t1 = time.time()
            print("MPCKMeans+MinMax costs %2.3fs" % (t1-t0))
            break
        except EmptyClustersException:
            print('Clustering Error')
    while True:
        try:
            t0 = time.time()
            ARI_MPCKmeans_NPU = ARI_semi_active(X, y, K, nc, 'MPCKmeans', 'NPU')
            t1 = time.time()
            print("MPCKMeans+NPU costs %2.3fs" % (t1-t0))
            break
        except EmptyClustersException:
            print('Clustering Error')
    while True:
        try:
            A_MPCKmeans, y_fit = active_MPCKmeans(X, y, K, nc, nc_per_step)
            ARI_MPCKmeans_active = adjusted_rand_score(y, y_fit)
            break
        except EmptyClustersException:
            print('Clustering Error')   
    while True:
        try:
            t10 = time.time()
            A_active, S_active, D_active = active_metric_learn(X, y, K, nc, nc_per_step, query_method='active', include_H=True, true_H=true_H)
            X_proj = X.dot(mat_sqrt(A_active))
            ARI_active_PCKmeans = ARI_clustering(X_proj, y, K, 'PCKmeans', S_active, D_active)
            t11 = time.time()
            break
        except EmptyClustersException:
            print('Clustering Error')
    while True:
        try:
            t20 = time.time()
            A_random, S_random, D_random = active_metric_learn(X, y, K, nc, nc_per_step, query_method='random', true_H=true_H)
            X_proj = X.dot(mat_sqrt(A_random))
            ARI_random_PCKmeans = ARI_clustering(X_proj, y, K, 'PCKmeans', S_random, D_random)
            t21 = time.time()
            break
        except EmptyClustersException:
            print('Clustering Error')
    while True:
        try:
            A_random_without_H, S_random, D_random = active_metric_learn(X, y, K, nc, nc_per_step, query_method='random', true_H=true_H, include_H=False)
            X_proj = X.dot(mat_sqrt(A_random_without_H))
            ARI_random_PCKmeans_without_H = ARI_clustering(X_proj, y, K, 'PCKmeans', S_random, D_random)
            break
        except EmptyClustersException:
            print('Clustering Error')
    # MPCKmeans with single diagonal metric matrix
    while True:
        try:
            t30 = time.time()
            ARI_MPCKmeans = ARI_clustering(X, y, K, 'MPCKmeans', S_random, D_random)
            t31 = time.time()
            break
        except EmptyClustersException:
            print('Clustering Error')
                 
    while True:
        try:
            ARI_PCKmeans = ARI_clustering(X, y, K, 'PCKmeans', S_random, D_random)
            break
        except EmptyClustersException:
            print('Clustering Error')
    # MPCKmeans with multiple full metric matrix
    while True:
        try:
            t40 = time.time()
            ARI_MPCKmeansMF = ARI_clustering(X, y, K, 'MPCKmeansMF', S_random, D_random)
            t41 = time.time()
            break
        except EmptyClustersException:
            print('Clustering Error')    
    print('ARI_active_PCKmeans=%2.3f, ARI_random_PCKmeans=%2.3f, ARI_MPCKmeans=%2.3f, ARI_MPCKmeansMF=%2.3f, ARI_MPCKmeans_active=%2.3f,  ARI_PCKmeans=%2.3f, ARI_random_PCKmeans_without_H=%2.3f' % (ARI_active_PCKmeans, ARI_random_PCKmeans, ARI_MPCKmeans, ARI_MPCKmeans_active, ARI_MPCKmeansMF, ARI_PCKmeans, ARI_random_PCKmeans_without_H))
    print('proposed_active costs %2.3fs, proposed_random costs %2.3fs, MPCKmeans costs %2.3fs, MPCKmeansMF costs %2.3fs' %(t11-t10, t21-t20, t31-t30, t41-t40))
    print('ARI_PCKmeans_MinMax=%2.3f, ARI_PCKmeans_NPU=%2.3f, ARI_MPCKmeans_Minmax=%2.3f, ARI_MPCKmeans_NPU=%2.3f' % (ARI_PCKmeans_MinMax, ARI_PCKmeans_NPU, ARI_MPCKmeans_MinMax, ARI_MPCKmeans_NPU))
    return  ARI_active_PCKmeans, ARI_random_PCKmeans, ARI_MPCKmeans, ARI_MPCKmeansMF, ARI_MPCKmeans_active, ARI_PCKmeans, ARI_random_PCKmeans_without_H, ARI_PCKmeans_MinMax, ARI_PCKmeans_NPU, ARI_MPCKmeans_MinMax, ARI_MPCKmeans_NPU

from joblib import Parallel, delayed, parallel_backend
import sys
if __name__ == '__main__':
    np.random.seed(1)
    reps = 10 
    num_constraints = range(20, 200, 40)
    nc_per_step = 10
    P1 = 10
    P2 = 30
    result_ARI_active_PCKmeans = np.zeros((reps, len(num_constraints)))
    result_ARI_random_PCKmeans = np.zeros((reps, len(num_constraints)))
    result_ARI_MPCKmeans = np.zeros((reps, len(num_constraints)))
    result_ARI_MPCKmeansMF= np.zeros((reps, len(num_constraints)))
    result_ARI_MPCKmeans_active = np.zeros((reps, len(num_constraints)))
    result_ARI_PCKmeans = np.zeros((reps, len(num_constraints)))
    result_ARI_PCKmeans_MinMax = np.zeros((reps, len(num_constraints)))
    result_ARI_PCKmeans_NPU = np.zeros((reps, len(num_constraints)))
    result_ARI_MPCKmeans_MinMax = np.zeros((reps, len(num_constraints)))
    result_ARI_MPCKmeans_NPU = np.zeros((reps, len(num_constraints)))
    result_ARI_random_PCKmeans_without_H = np.zeros((reps, len(num_constraints)))
    
    n_super_instances = range(10, 110, 10)
    result_COBRA = [[0 for _ in range(len(n_super_instances))] for _ in range(reps)]
    
    parallel = True
    if not parallel:  
        for idx, nc in enumerate(n_super_instances):
            for rep in range(reps):
                n_query, ARI = repeat_COBRA(nc, rep, P1, P2)
                result_COBRA[rep][idx] = (n_query, ARI)
                
        for idx, nc in enumerate(num_constraints):
            for rep in range(reps):
                # active         
                ARI_active_PCKmeans, ARI_random_PCKmeans, ARI_MPCKmeans, ARI_MPCKmeansMF, ARI_MPCKmeans_active, ARI_PCKmeans, ARI_random_PCKmeans_without_H, ARI_PCKmeans_MinMax, ARI_PCKmeans_NPU, ARI_MPCKmeans_MinMax, ARI_MPCKmeans_NPU = comparison_high_dim(nc, rep, P1, P2, nc_per_step)
                
                result_ARI_active_PCKmeans[rep, idx] = ARI_active_PCKmeans
                result_ARI_random_PCKmeans[rep, idx] = ARI_random_PCKmeans
                result_ARI_MPCKmeans[rep, idx] = ARI_MPCKmeans
                result_ARI_MPCKmeans_active[rep, idx] = ARI_MPCKmeans_active
                result_ARI_PCKmeans[rep, idx] = ARI_PCKmeans
                result_ARI_MPCKmeansMF[rep, idx] = ARI_MPCKmeansMF
                result_ARI_random_PCKmeans_without_H[rep, idx] = ARI_random_PCKmeans_without_H
                result_ARI_PCKmeans_MinMax[rep, idx] = ARI_PCKmeans_MinMax
                result_ARI_PCKmeans_NPU[rep, idx] = ARI_PCKmeans_NPU
                result_ARI_MPCKmeans_MinMax[rep, idx] = ARI_MPCKmeans_MinMax
                result_ARI_MPCKmeans_NPU[rep, idx] = ARI_MPCKmeans_NPU
    else:

        idx_set_COBRA = [(i,j) for i in n_super_instances for j in range(reps)]
        result_C =  Parallel(n_jobs=4, verbose=10)(delayed(repeat_COBRA)(nc, rep, P1, P2) for nc, rep in idx_set_COBRA)
        ns_idx = dict()
        for idx, ns in enumerate(n_super_instances):
            ns_idx[ns] = idx
        for d1, d2 in enumerate(idx_set_COBRA):
            ns, rep = d2
            idx = ns_idx[ns]
            result_COBRA[rep][idx] = result_C[d1]
            
        save_path = './active_vs_random/infer_S_D_from_H_fuzzy_learn_A_simulation_diag/10_per_cluster/mu_5/P1_'+str(P1)+'P2_'+str(P2)+'/incremental_'+str(nc_per_step)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        np.save(os.path.join(save_path, 'result_COBRA.npy'), result_COBRA)
        idx_set = [(i,j) for i in num_constraints for j in range(reps)]
        
        n_query = np.array([[ result_COBRA[j][i][0] for i in range(len(n_super_instances))] for j in range(reps)])
        ARI_COBRA = np.array([[ result_COBRA[j][i][1] for i in range(len(n_super_instances))] for j in range(reps)])
        with open(os.path.join(save_path,'result_COBRA.tsv'), 'w+') as file:
            file.write('number of super instances')
            for n in (n_super_instances):
                file.write('\t' + str(n))
            output(file, n_query, 'average number of queries')
            output(file, ARI_COBRA, 'average ARI of COBRA')
        
        result = Parallel(n_jobs=4, verbose=10)(delayed(comparison_high_dim)(nc, rep, P1, P2, nc_per_step) for nc, rep in idx_set)
        
        nc_idx = dict()
        for idx, nc in enumerate(num_constraints):
            nc_idx[nc] = idx
        for d1, d2 in enumerate(idx_set):
            nc, rep = d2 
            idx = nc_idx[nc]
            result_ARI_active_PCKmeans[rep, idx], result_ARI_random_PCKmeans[rep, idx],  result_ARI_MPCKmeans[rep, idx], result_ARI_MPCKmeansMF[rep, idx], result_ARI_MPCKmeans_active[rep, idx], result_ARI_PCKmeans[rep, idx], result_ARI_random_PCKmeans_without_H[rep,idx], result_ARI_PCKmeans_MinMax[rep, idx], result_ARI_PCKmeans_NPU[rep, idx], result_ARI_MPCKmeans_MinMax[rep, idx], result_ARI_MPCKmeans_NPU[rep, idx] = result[d1]
    
   
    np.savez(os.path.join(save_path, 'result.npz'), result_ARI_active_PCKmeans=result_ARI_active_PCKmeans,
result_ARI_random_PCKmeans=result_ARI_random_PCKmeans,
result_ARI_MPCKmeans=result_ARI_MPCKmeans,
result_ARI_MPCKmeans_active=result_ARI_MPCKmeans_active,
result_ARI_MPCKmeansMF=result_ARI_MPCKmeansMF,
result_ARI_PCKmeans=result_ARI_PCKmeans,
result_ARI_random_PCKmeans_without_H=result_ARI_random_PCKmeans_without_H,
result_ARI_PCKmeans_MinMax=result_ARI_PCKmeans_MinMax,
result_ARI_PCKmeans_NPU=result_ARI_PCKmeans_NPU,
result_ARI_MPCKmeans_MinMax=result_ARI_MPCKmeans_MinMax,
result_ARI_MPCKmeans_NPU=result_ARI_MPCKmeans_NPU
)
    def save_figure(result, name, num_constraints):
        fig, ax = plt.subplots()
        ax.boxplot(result_ARI_active_PCKmeans)
        ax.set_xticklabels(num_constraints)
        ax.set_xlabel('number of constraints')
        ax.set_ylabel(name)
        ax.set_ylim(0,1.05)
        fig.savefig(os.path.join(save_path,name+'.png'))
        
    fig, ax = plt.subplots()
    ax.boxplot(result_ARI_active_PCKmeans)
    ax.set_xticklabels(num_constraints)
    ax.set_xlabel('number of constraints')
    ax.set_ylabel('ARI_active_PCKmeans')
    ax.set_ylim(0,1.05)
    fig.savefig(os.path.join(save_path,'ARI_active_PCKmeans.png'))    
  
    fig, ax = plt.subplots()
    ax.boxplot(result_ARI_random_PCKmeans)
    ax.set_xticklabels(num_constraints)
    ax.set_xlabel('number of constraints')
    ax.set_ylabel('ARI_random_PCKmeans')  
    ax.set_ylim(0,1.05)
    fig.savefig(os.path.join(save_path,'ARI_random_PCKmeans.png'))    
    
    fig, ax = plt.subplots()
    ax.boxplot(result_ARI_MPCKmeans)
    ax.set_xticklabels(num_constraints)
    ax.set_xlabel('number of constraints')
    ax.set_ylabel('ARI_MPCKmeans')  
    ax.set_ylim(0,1.05)
    fig.savefig(os.path.join(save_path,'ARI_MPCKmeans.png')) 
    
    fig, ax = plt.subplots()
    ax.boxplot(result_ARI_MPCKmeans_active)
    ax.set_xticklabels(num_constraints)
    ax.set_xlabel('number of constraints')
    ax.set_ylabel('ARI_MPCKmeans_active')  
    ax.set_ylim(0,1.05)
    fig.savefig(os.path.join(save_path,'ARI_MPCKmeans_active.png')) 
    
    fig, ax = plt.subplots()
    ax.boxplot(result_ARI_MPCKmeansMF)
    ax.set_xticklabels(num_constraints)
    ax.set_xlabel('number of constraints')
    ax.set_ylabel('ARI_MPCKmeansMF') 
    ax.set_ylim(0,1.05)
    fig.savefig(os.path.join(save_path,'ARI_MPCKmeansMF.png')) 
    
    fig, ax = plt.subplots()
    ax.boxplot(result_ARI_PCKmeans)
    ax.set_xticklabels(num_constraints)
    ax.set_xlabel('number of constraints')
    ax.set_ylabel('ARI_PCKmeans')  
    ax.set_ylim(0,1.05)
    fig.savefig(os.path.join(save_path,'ARI_PCKmeans.png')) 
    
    save_figure(result_ARI_random_PCKmeans_without_H, 'ARI_random_PCKmeans_without_H', num_constraints)
    save_figure(result_ARI_PCKmeans_MinMax, 'ARI_PCKmeans_MinMax', num_constraints)
    save_figure(result_ARI_PCKmeans_NPU, 'ARI_PCKmeans_NPU', num_constraints)
    save_figure(result_ARI_MPCKmeans_MinMax, 'ARI_MPCKmeans_MinMax', num_constraints)
    save_figure(result_ARI_MPCKmeans_NPU, 'ARI_MPCKmeans_NPU', num_constraints)
    
    def output(file, result, setting_name):
        file.write('\n'+setting_name)
        res_mean = np.mean(result, 0)
        res_sd = np.std(result, 0)
        for n in range(len(res_mean)):
            file.write('\t %2.3f(%2.3f)' % (res_mean[n], res_sd[n]))
        
        
    with open(os.path.join(save_path,'result.tsv'), 'w+') as file:
        file.write('\t')
        for nc in num_constraints:
            file.write("%d\t" % nc)
        output(file, result_ARI_active_PCKmeans, 'ARI_active_PCKmeans')
        output(file, result_ARI_random_PCKmeans, 'ARI_random_PCKmeans')
        output(file, result_ARI_MPCKmeans, 'ARI_MPCKmeans')
        output(file, result_ARI_MPCKmeansMF, 'ARI_MPCKmeansMF')
        output(file, result_ARI_MPCKmeans_active, 'ARI_MPCKmeans_active')
        output(file, result_ARI_PCKmeans, 'ARI_PCKmeans')       
        output(file, result_ARI_random_PCKmeans_without_H, 'ARI_random_PCKmeans_without_H')
        output(file, result_ARI_PCKmeans_MinMax, 'ARI_PCKmeans_MinMax')       
        output(file, result_ARI_PCKmeans_NPU, 'ARI_PCKmeans_NPU')       
        output(file, result_ARI_MPCKmeans_MinMax, 'ARI_MPCKmeans_MinMax')       
        output(file, result_ARI_MPCKmeans_NPU, 'ARI_MPCKmeans_NPU')       
        

   
