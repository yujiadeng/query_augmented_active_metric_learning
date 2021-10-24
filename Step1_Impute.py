# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 17:25:14 2019

@author: Yujia Deng

Step 1. Impute membership H given the label set with multidirectional penalty

H_hat = argmin sum_{y_{ij}\neq 0} (y_{ij}-H_i^TH_j)^2+\lambda sum_{i,k} min(|H_{ik}|, |H_{ik}-1|)

s.t. 
1^T H_i = 1
H_{i,k} \geq 0
"""
import numpy as np
from numpy.linalg import inv
from helper import *
from mosek.fusion import *
# import cvxpy as cp

def soft_threshold(lambd, y):
    return(np.sign(y)*np.max((np.abs(y)-lambd,0)))
    
def infer_membership_from_label(S:'similar set', D:'dissimilar set', N:'number of samples', K:'number of clusters', lambd: 'multidirectional penalty weight'=1, rho:'multiplier weight'=10, eps=1e-3,inspect=False) -> "N by K array":
#    H = cp.Variable((N, K))
#    sum_SD = 0
#    for (i, j) in S:
#        sum_SD += (1 - H[i,:]@ H[j,:])**2
#    for (i, j) in D:
#        sum_SD += (H[i,:] @ H[j,:])**2
#    objective = cp.Minimize(sum_SD)
#    constraints = [0 <= H, H @ np.repeat(1,K)==np.repeat(1,N)]
#    prob = cp.Problem(objective, constraints)
#    prob.solve(verbose=True, gp=True)
#    return H.value
#    
    # Initialization
#    H = np.repeat(1/K, N*K).reshape((N, K))
    nc = len(S) + len(D)
    inits = 5
    L_past = np.infty
    for init in range(inits): # try different initiations to avoid loacal extrema
        H = np.random.randn(N,K)
    #    H = np.zeros((N,K))
    #    for k in range(K):
    #        H[int(np.random.choice(range(K),1))] = 1
        Z = H.copy()
        v = np.repeat(0.0, N*K).reshape((N, K))
        w = np.repeat(0.0, N)
        u = np.repeat(0.0, N*K).reshape((N, K))
        
        iters = 20
        sets_j = [dict()]*N
        for i in range(N):
            set_j = dict()
            for (i_, j) in D:
                if i_ == i:
                    set_j[j] = 0
                elif j == i:
                    set_j[i_] = 0
            for (i_, j) in S:
                if i_ == i:
                    set_j[j] = +1
                elif j == i:
                    set_j[i_] = +1
            sets_j[i] = set_j
        def loss(sets_j, H, Z, u, v, w, lambd, rho):
            L = lambd * np.sum(np.min((np.abs(Z), np.abs(Z-1)))) + \
                rho/2 * np.sum((H-Z)**2) + \
                rho/2 * np.sum((np.inner(H, np.repeat(1,K))-np.repeat(1,N))**2) + \
                1/2/rho * np.sum(np.max((u*0, (u-rho*H)))**2 - u**2)
            for i in range(H.shape[0]):
                tmp = 0
                set_j = sets_j[i]
                for j in set_j:
                    tmp += 1/2*(set_j[j]-np.inner(H[i,:],H[j,:]))**2
                L += np.inner(v[i,:], H[i,:]-Z[i,:]) + \
                     w[i]*(np.inner(H[i,:], np.repeat(1,K))-1) + \
                     tmp
            return L
        
        L0 = loss(sets_j, H, Z, u, v, w, lambd, rho)
        for iter in range(iters):
#            print(loss(sets_j, H, Z, u, v, w, lambd, rho))
            # Step 1: update membership matrix H (row by row)
            reps = 20
            H_hat = H.copy()
            L1 = loss(sets_j, H, Z, u, v, w, lambd, rho)
            for rep in range(reps):     
                for i in range(N):
                    set_j = sets_j[i] # pairs involving i, value=1 if similar, 0 if dissimilar           
                    LHS = rho * (np.eye(K) + np.repeat(1, K*K).reshape((K,K))) # LHS without correction
                    RHS = rho * (Z[i,:] + np.repeat(1, K)) - v[i,:] - np.repeat(w[i], K) # RHS without correction
                    for j in set_j:
                        LHS += np.outer(H[j,:], H[j,:])
                        RHS += set_j[j]*H[j,:]
                    H_i_tilde = inv(LHS).dot(RHS)
                    A_i = np.diag([0 if H_i_tilde[k] > u[i,k]/rho else rho for k in range(K)])
                    b_i = [0 if H_i_tilde[k] > u[i,k]/rho else u[i,k] for k in range(K)]
                    H_i_hat = inv(LHS + A_i).dot(RHS + b_i)
                    H_hat[i,:] = H_i_hat
                L = loss(sets_j, H_hat, Z, u, v, w, lambd, rho)
                rel_rate_1 = abs(L-L1)/L1
                L1 = L
                if rel_rate_1 < eps:
#                    print('rel_rate_1=%2.3f' % rel_rate_1)
                    break
#            print('rep=%d, rel_rate_1=%2.3f' %(rep,rel_rate_1))
            H = H_hat.copy()                   
            # Step 2: update augmented matrix Z
            for i in range(N):
    #            if i<10:
    #                Z[i,0] = soft_threshold(lambd/rho, v[i,0]/rho + H[i,0] -1) + 1
    #                Z[i,1] = soft_threshold(lambd/rho, v[i,1]/rho + H[i,1]) 
    #                Z[i,2] = soft_threshold(lambd/rho, v[i,2]/rho + H[i,2]) 
    #            elif i<20:
    #                Z[i,1] = soft_threshold(lambd/rho, v[i,1]/rho + H[i,1] -1) + 1
    #                Z[i,0] = soft_threshold(lambd/rho, v[i,0]/rho + H[i,0]) 
    #                Z[i,2] = soft_threshold(lambd/rho, v[i,2]/rho + H[i,2])
    #            else:
    #                Z[i,2] = soft_threshold(lambd/rho, v[i,1]/rho + H[i,2] -1) + 1
    #                Z[i,0] = soft_threshold(lambd/rho, v[i,0]/rho + H[i,0]) 
    #                Z[i,1] = soft_threshold(lambd/rho, v[i,1]/rho + H[i,1])
#                Z_i_tilde = H[i,:] + v[i,:]/rho
#                Z[i,:] = [soft_threshold(lambd/rho, v[i,k]/rho + H[i,k]) if Z_i_tilde[k] < 1/2 else soft_threshold(lambd/rho, v[i,k]/rho + H[i,k] -1) + 1 for k in range(K)]
                
                Z[i,:] = H[i,:] + v[i,:]/rho
            # Step 3: udpate Lagrange multiplier u,v,w
            v += rho*(H - Z)
            w += rho*(np.inner(H, np.repeat(1,K)) - np.repeat(1,N))
            u = np.array([max(x, 0) for x in (u-rho*H).flat]).reshape(N,K)
            L = loss(sets_j, H, Z, u, v, w, lambd, rho)
            rel_rate = abs(L-L0)/L0
            L0 = L
            if  rel_rate < eps:
                break
        if L < L_past:
            ans = H.copy(), Z.copy(), L.copy(), v.copy(), w.copy(), u.copy()
            L_past = L
#            H_ans = H       
#        print('init=%d, L=%2.3f, rel_rate=%2.3f, L_past=%2.3f' % (init, L, rel_rate, L_past))
    if inspect:
        return ans
    else:
        return ans[0]

from itertools import combinations
from random import sample
from helper import *
# import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score

def test_case1():
    X, y, K, num_per_class, scale = load_high_dim3(3,3,10,seed=1)
    N = X.shape[0]
    U = set(combinations(range(N),2))
    S = set()
    D = set()
    nc= 10
    while len(S) == 0 or len(D) == 0:
        U = set(combinations(range(N),2))
        S = set()
        D = set()
        S, D, U = add_query(y, S, D, U, nc)
    H = infer_membership_from_label(S, D, N, K)
    return H
    
def test_case2():  
    np.random.seed(1)   
    reps = 30 
    num_constraints = range(20, 200, 10)
    result_loss = np.zeros((reps, len(num_constraints)))
    result_ARI = np.zeros((reps, len(num_constraints)))
    for idx, nc in enumerate(num_constraints):
        for rep in range(reps):
            print('rep=%d, number of constraints= %d' % (rep, nc))
            X, y, K, num_per_class, scale = load_high_dim3(3,3,10,seed=rep)
            N = X.shape[0]
            U = set(combinations(range(N),2))
            S = set()
            D = set()
            while len(S) == 0 or len(D) == 0:
                U = set(combinations(range(N),2))
                S = set()
                D = set()
                S, D, U = add_query(y, S, D, U, nc)
            H, Z, L, v, w, u = infer_membership_from_label(S, D, N, K, lambd=1, inspect=True) 
#            H = infer_membership_from_label(S, D, N, K, lambd=1, ins) 
            H = np.array([0 if x< 1e-3 else 1 if x>1 else x for x in H.flat]).reshape(H.shape)
            y_hat = np.argmax(H,1)# estimated label
            result_loss[rep, idx] = L
            ARI = adjusted_rand_score(y, y_hat)
            result_ARI[rep, idx] = ARI
            print('Loss=%2.3f, ARI=%2.3f' % (L, ARI))
    # plot
    np.savez('step1_result_without_penalty', result_loss=result_loss, result_ARI=result_ARI)
    fig, ax = plt.subplots()
    ax.boxplot(result_loss)
    ax.set_xticklabels(num_constraints)
    ax.set_xlabel('number of constraints')
    ax.set_ylabel('loss func')
    fig.savefig('loss_without_penalty.png')
    
    fig, ax = plt.subplots()
    ax.boxplot(result_ARI)
    ax.set_xticklabels(num_constraints)
    ax.set_ylabel('ARI')
    fig.savefig('ARI_without_penalty.png')


if __name__ == '__main__':
    H = test_case1()
    
    
    
            
    
    
    