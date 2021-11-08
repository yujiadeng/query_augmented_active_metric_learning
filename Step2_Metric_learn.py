# -*- coding: utf-8 -*-
"""
Step 2: update A = MM^T through SDP

@author: Yujia Deng
"""

import numpy as np
import mosek
from  mosek.fusion import *
import cvxpy as cp
from active_semi_clustering.exceptions import EmptyClustersException

def metric_learn_A(X:'N by p data matrix', S:'similar set', D:'dissimilar set', H:'infered matrix from step 1', include_H = True, diag=True, lambd=0, gamma=0):
    N, p = X.shape
    
    denom = [x if x>0 else 1 for x in np.std(X,0)]
    X = X / denom
    
    M_D = np.zeros((p,p))
    for i, j in D:
        M_D += np.outer(X[i,:]-X[j,:], X[i,:]-X[j,:])
    
    U = set(combinations(range(N),2))
    
    K = H.shape[1]
    # penality on the distance to the center
    tmp = np.sum(H*H, 0)
    center_mu = np.zeros((K, p))
    for k in range(K):
        for j in range(p):
            center_mu[k,j] = np.sum([H[i,k]**2*X[i,j] for i in range(N)] ) / tmp[k]
    
    center_vec = np.zeros(p)
    for j in range(p):
        center_vec[j] = sum([ H[i,k]**2 * (X[i,j] - center_mu[k,j])**2 for i in range(N) for k in range(K)])
       
    if diag:
        a = cp.Variable(p)
        a.value = np.repeat(1,p) #initial
        center_sum = center_vec @ a
        M_S = np.zeros(p)
        for i, j in S:
            M_S += (X[i] - X[j])**2
        M_D = np.zeros(p)
        for i, j in D:
            M_D += np.sqrt((X[i] - X[j]) ** 2)
                  
        M_W_S = np.zeros(p)
        M_W_D = np.zeros(p)
        for i, j in U:
            coef = 2 * (H[i].dot(H[j])) - 1
            coef_S = max(coef, 0)
            coef_D = max(-coef, 0)
            M_W_S += (X[i] - X[j])**2 * coef_S
            M_W_D += np.sqrt((X[i] - X[j])* (X[i] - X[j]) * coef_D)
        sum_S = M_S @ a
        sum_D = M_D @ cp.sqrt(a)
        sum_W_S = M_W_S @ a
        sum_W_D = M_W_D @ cp.sqrt(a)
        if include_H:
            objective = cp.Minimize(2 * lambd *sum_S + 2 * lambd* sum_W_S - cp.log(sum_D) - cp.log(sum_W_D) +  center_sum + gamma*cp.pnorm(a, p=2)**2)
        else:
            objective = cp.Minimize(sum_S - cp.log(sum_D))
        constraints = [0 <= a]
        prob = cp.Problem(objective, constraints)
        prob.solve(warm_start=True, verbose=False)

        print('lambd=%d, sum(abs(a))=%2.3f'% ( lambd, np.abs(a.value)@np.repeat(1,p)))
        print('Diagonal entries of the proposed method')
        ans = np.diag(a.value/np.linalg.norm(a.value))
        print(np.diag(ans))
        return ans
    else:
        print('Full rank A')
        A = cp.Variable((p, p), PSD=True)
        M_S = np.zeros((p,p))
        for i, j in S:
            M_S += np.outer(X[i,:]-X[j,:], X[i,:]-X[j,:])
      
        W = np.dot(H, H.T)
        L = np.diag(np.sum(W, axis=1))
        M_W = np.dot(np.dot(X.T, L - W), X)
        
        M_mu = np.zeros((p, p))
        for i in range(N):
            for k in range(K):
                M_mu += H[i,k]**2* np.outer(X[i, :] - center_mu[k, :], X[i, :] - center_mu[k, :])
        
        ob = 0
        for (i, j) in D:
            ob += cp.sqrt( ((X[i,:] - X[j,:]) @ A) @ (X[i,:] - X[j,:])/ len(D)) - lambd* cp.trace(A)
            
        objective = cp.Maximize(ob)
        constraints = [cp.trace((M_S/len(S) + M_W/N**2 + M_mu/N/K)@A) <= 1]
        prob = cp.Problem(objective, constraints)
        prob.solve(warm_start=False, verbose=False)
        return A.value

def metric_learn_backward(P2:"number of noisy dimensions", X0, S, D, H):
    """
    sequentially exclude the most noisy dimensions
    return: projection matrix P
    A = P.dot(P)
    """
    X = X0.copy()
    final_P = np.eye(X.shape[1])   
    for d in range(P2):
        A = metric_learn_A(X, S, D, H)
        P = proj_mat(A)
        X = np.dot(X, np.eye(p) - P)
        final_P = final_P.dot(np.eye(p) - P)
    return final_P

from helper import *  
from itertools import combinations   

if __name__ == '__main__':
    np.random.seed(1)   
    reps = 30 
    num_constraints = range(20, 200, 10)
    result_ARI = np.zeros((reps, len(num_constraints)))
    result_ARI_Kmeans = np.zeros((reps, len(num_constraints)))
    result_ARI_PCKmeans = np.zeros((reps, len(num_constraints)))
    result_ARI_MPCKmeans = np.zeros((reps, len(num_constraints)))
    for idx, nc in enumerate(num_constraints):
        for rep in range(reps):
            print('rep=%d, number of constraints= %d' % (rep, nc))
            P1 = 6 # true space dimension
            P2 = 3 # noisy space dimension
            X0, y, K, num_per_class, scale = load_high_dim3(P1,P2,10,seed=rep, random_scale=False)
            X = X0
            p = X.shape[1]
            N = X.shape[0]
            N = X.shape[0]
            U = set(combinations(range(N),2))
            S = set()
            D = set()
            while len(S) < 5 or len(D) <5:
                U = set(combinations(range(N),2))
                S = set()
                D = set()
                S, D, U = add_query(y, S, D, U, nc)
            H = np.zeros((N, K))
            for n in range(N):
                H[n, y[n]] = 1
            A = metric_learn_A(X, S, D, H, diag=False)
            X_proj = X.dot(mat_sqrt(A))
            while True:
                try:
                    ARI = ARI_clustering(X_proj, y, P1, 'PCKmeans', S, D)
                    break
                except EmptyClustersException:
                    print('Empty cluster.')
            ARI_Kmeans = ARI_clustering(X0, y, P1, 'kmeans', S, D)
            while True:
                try:
                    ARI_PCKmeans = ARI_clustering(X0, y, P1, 'PCKmeans', S, D)
                    break 
                except EmptyClustersException:
                    print('Empty cluster.')
            while True:
                try:
                    ARI_MPCKmeans = ARI_clustering(X0, y, P1, 'MPCKmeans', S, D)
                    break
                except EmptyClustersException:
                    print('Empty cluster.')
            result_ARI[rep, idx] = ARI
            result_ARI_Kmeans[rep, idx] = ARI_Kmeans
            result_ARI_PCKmeans[rep, idx] = ARI_PCKmeans
            result_ARI_MPCKmeans[rep, idx] = ARI_MPCKmeans
            print('ARI_Kmeans= %2.3f, ARI_PCKmeans=%2.3f, ARI_MPCKmeans=%2.3f, ARI=%2.3f' % (ARI_Kmeans, ARI_PCKmeans, ARI_MPCKmeans, ARI))
    
    fig, ax = plt.subplots()
    ax.boxplot(result_ARI)
    ax.set_xticklabels(num_constraints)
    ax.set_xlabel('number of constraints')
    ax.set_ylabel('ARI')
    
    fig, ax = plt.subplots()
    ax.boxplot(result_ARI_Kmeans)
    ax.set_xticklabels(num_constraints)
    ax.set_xlabel('number of constraints')
    ax.set_ylabel('ARI_Kmeans')
    
    fig, ax = plt.subplots()
    ax.boxplot(result_ARI_PCKmeans)
    ax.set_xticklabels(num_constraints)
    ax.set_xlabel('number of constraints')
    ax.set_ylabel('ARI_PCKmeans')