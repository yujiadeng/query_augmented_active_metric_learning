# -*- coding: utf-8 -*-
"""
Compare the ARI w/ and w/o query augmentation. Generate Figure 1 and Figure 2

@author: Yujia Deng
"""
from Step2_Metric_learn import *
from Step1_Impute import infer_membership_from_label
import cvxpy as cp
from helper import plot3D

def metric_learn_A_diag(X, S, D, H, include_H=True, diag=True):
    N, p = X.shape
    M_S = np.zeros((p,p))
    for i, j in S:
        M_S += np.outer(X[i,:]-X[j,:], X[i,:]-X[j,:])
    M_D = np.zeros((p,p))
    for i, j in D:
        M_D += np.outer(X[i,:]-X[j,:], X[i,:]-X[j,:])
    W = np.dot(H, H.T)
    L = np.diag(np.sum(W, axis=1))
    M_W = np.dot(np.dot(X.T, L - W), X)
    
    a = cp.Variable(p)
    a.value = np.repeat(1,p) #initial
    sum_S = 0
    for (i, j) in S:
        sum_S += (X[i]-X[j])**2 @a
    sum_D = 0
    for (i, j) in D:
        sum_D += cp.sqrt((X[i]-X[j])**2 @a)
    objective = cp.Minimize(sum_S - cp.log(sum_D) + np.diag(M_W)@a)
    constraints = [0 <= a]
    prob = cp.Problem(objective, constraints)
    prob.solve(warm_start=True)
    return np.diag(a.value/np.linalg.norm(a.value))
    

def metric_learn_A(X:'N by p data matrix', S:'similar set', D:'dissimilar set', H:'infered matrix from step 1', include_H = True, diag=True):
    N, p = X.shape
    M_S = np.zeros((p,p))
    for i, j in S:
        M_S += np.outer(X[i,:]-X[j,:], X[i,:]-X[j,:])
    M_D = np.zeros((p,p))
    for i, j in D:
        M_D += np.outer(X[i,:]-X[j,:], X[i,:]-X[j,:])
    W = np.dot(H, H.T)
    L = np.diag(np.sum(W, axis=1))
    M_W = np.dot(np.dot(X.T, L - W), X)
    
    if diag:
        a = cp.Variable(p)
        a.value = np.repeat(1,p) #initial
        sum_S = 0
        for (i, j) in S:
            sum_S += (X[i]-X[j])**2 @a
        sum_D = 0
        for (i, j) in D:
            sum_D += cp.sqrt((X[i]-X[j])**2 @a)
        if include_H:
            objective = cp.Minimize(sum_S - cp.log(sum_D) + np.diag(M_W)@a)
        else:
            objective = cp.Minimize(sum_S - cp.log(sum_D))
        constraints = [0 <= a]
        prob = cp.Problem(objective, constraints)
        prob.solve(warm_start=True)
        return np.diag(a.value/np.linalg.norm(a.value))
    else:
        pass
   
def metric_learn_backward(P2:"number of noisy dimensions", X0, S, D, H, include_H = True, diag=True):
    """
    sequentially exclude the most noisy dimensions
    return: projection matrix P
    A = P.dot(P)
    """
    X = X0.copy()
    final_P = np.eye(X.shape[1])   
    if diag:
        A = metric_learn_A(X, S, D, H, include_H, diag)
        P = mat_sqrt(A)
        return P
    else:
        for d in range(P2):
            try:
                A = metric_learn_A(X, S, D, H, include_H, diag)
            except SolutionError:
                print('SolverError. Return current result')
                return final_P
            P = proj_mat(A)
            X = np.dot(X, np.eye(p) - P)
            final_P = final_P.dot(np.eye(p) - P) # only rotate, no scaling     
    return final_P

from helper import *  
from itertools import combinations   

def ratio_S_D(X, S, D):
    """
    calculate the ratio of summatin of distance in S and D
    """
    d_S = sum(np.inner(X[i,:] - X[j,:], X[i,:] - X[j,:]) for (i, j) in S)
    d_D = sum(np.inner(X[i,:] - X[j,:], X[i,:] - X[j,:]) for (i, j) in D)
    return d_S / d_D

if __name__ == '__main__':
    np.random.seed(1)   
    reps = 30
    num_constraints = range(10, 110, 10)
    result_ARI_H = np.zeros((reps, len(num_constraints)))
    result_ARI_without_H = np.zeros((reps, len(num_constraints)))
    result_ratio_H = np.zeros((reps, len(num_constraints)))
    result_ratio_without_H = np.zeros((reps, len(num_constraints)))
    for idx, nc in enumerate(num_constraints):
        for rep in range(reps):
            print('rep=%d, number of constraints= %d' % (rep, nc))
            P1 = 6 # true space dimension
            P2 = 3 # noisy space dimension
            X0, y, K, num_per_class, scale = load_high_dim3(P1,P2,10,seed=rep, random_scale=False)
            X = X0
            p = X.shape[1]
            N = X.shape[0]
            U = set(combinations(range(N),2))
            S = set()
            D = set()
            while len(S) < 1 or len(D) < 1:
                U = set(combinations(range(N),2))
                S = set()
                D = set()
                S, D, U = add_query(y, S, D, U, nc)
            

            ## inferred from step 1
            H = infer_membership_from_label(S, D, N, K)
            
            P_H = metric_learn_backward(P2, X, S, D, H, include_H=True)
            X_proj_H = X.dot(P_H)          
            P_without_H = metric_learn_backward(P2, X, S, D, H,include_H=False)
            X_proj_without_H = X.dot(P_without_H)
            while True:
                try:
                    ARI_H = ARI_clustering(X_proj_H, y, K, 'Kmeans', S, D)
                    break
                except:
                    print('Empty cluster.')
                    
            while True:
                try:
                    ARI_without_H = ARI_clustering(X_proj_without_H, y, K, 'Kmeans', S, D)
                    break
                except:
                    print('Empty cluster.')
        
            ratio_S_D_H = ratio_S_D(X_proj_H, S, D)
            ratio_S_D_without_H = ratio_S_D(X_proj_without_H, S, D)
            
            result_ARI_H[rep, idx] = ARI_H
            result_ARI_without_H[rep, idx] = ARI_without_H
            result_ratio_H[rep, idx] = ratio_S_D_H
            result_ratio_without_H[rep, idx] = ratio_S_D_without_H
            print("ARI_H=%2.3f, ARI_without_H=%2.3f, ratio_S_D_H=%2.3f, ratio_S_D_without_H=%2.3f" % (ARI_H, ARI_without_H, ratio_S_D_H, ratio_S_D_without_H))
            
    np.savez('step2_comparison_H_without_H',
             result_ARI_H=result_ARI_H,
             result_ARI_without_H=result_ARI_without_H,
             ratio_S_D_H=ratio_S_D_H,
             ratio_S_D_without_H=ratio_S_D_without_H)
    
    import pandas as pd    
    import seaborn as sns
    import matplotlib.pyplot as plt
    data_H = pd.DataFrame(result_ARI_H, columns=range(10, 110, 10))
    data_H = data_H.melt(var_name='number_of_constraints', value_name='value' )
    data_H['Method'] = 'with inferred constraints'  


    data_no_H = pd.DataFrame(result_ARI_without_H, columns=range(10, 110, 10))
    data_no_H = data_no_H.melt(var_name='number_of_constraints', value_name='value' )    
    data_no_H['Method'] = 'without inferred constraints'  

    data = pd.concat([data_H, data_no_H])
    sns.set_style("whitegrid")

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(ax=ax, x='number_of_constraints', y='value', hue='Method', data=data, palette="Set3")
    ax.set_xlabel('Number of constraints')
    ax.set_ylabel('ARI')
    fig.savefig('../figure/Figure_2.png', bbox_inches='tight')

    X, y, K, num_per_class, scale = load_high_dim3(3,3,50,seed=1, random_scale=False)
    fig, ax = plot3D(X[:,0:3], y)
    fig.savefig('./figure/Figure_1_1.png')

    fig, ax = plot3D(X[:,3:6], y)
    fig.savefig('./figure/Figure_1_2.png')
    
    