# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 22:35:57 2019

@author: Yujia Deng

Step 2: update A = MM^T through SDP
"""

import numpy as np
import mosek
from  mosek.fusion import *
import cvxpy as cp
from active_semi_clustering.exceptions import EmptyClustersException
 
#def geometric_mean(M, x, t):
#    n = int(x.getSize())
#    if n==1:
#        M.constraint(Expr.sub(t, x), Domain.lessThan(0.0))
#    else:
#        t2 = M.variable()
#        M.constraint(Var.hstack(t2, x.index(n-1), t), Domain.inPPowerCone(1-1.0/n))
#        geometric_mean(M, x.slice(0,n-1), t2)
#
#def prod_greater_0(M, v, prev):
#    n = int(v.getSize())
#    if n == 1:
#        M.constraint(Expr.mul(prev, v), Domain.greaterThan(1.0))
#    else:
#        prev = Expr.dot(prev, v.index(n-1))
#        prod_greater_0(M, v.slice(0, n-1), prev)



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
    #    sum_D = sum(np.inner((X[i,:]-X[j,:])*a, (X[i,:]-X[j,:])) for (i, j) in D)
    #    objective = cp.Minimize(sum_S-cp.log(sum_D))
#        sum_S = 0
#        for (i, j) in S:
#            sum_S += (X[i]-X[j])**2 @a
#        sum_D = 0
#        for (i, j) in D:
#            sum_D += cp.sqrt((X[i]-X[j])**2 @a)
        
        center_sum = center_vec @ a
        M_S = np.zeros(p)
        for i, j in S:
            M_S += (X[i] - X[j])**2
        M_D = np.zeros(p)
        for i, j in D:
            M_D += np.sqrt((X[i] - X[j]) ** 2)
                  
        M_W_S = np.zeros(p)
        M_W_D = np.zeros(p)
#        sum_W_D = 0
        for i, j in U:
            coef = 2 * (H[i].dot(H[j])) - 1
            coef_S = max(coef, 0)
            coef_D = max(-coef, 0)
#            sum_W_S += (X[i] - X[j])**2 @a * coef_S
            M_W_S += (X[i] - X[j])**2 * coef_S
            M_W_D += np.sqrt((X[i] - X[j])* (X[i] - X[j]) * coef_D)
            
#            sum_W_D += cp.sqrt((X[i] - X[j])**2 @ a * coef_D)
        sum_S = M_S @ a
        sum_D = M_D @ cp.sqrt(a)
        sum_W_S = M_W_S @ a
        sum_W_D = M_W_D @ cp.sqrt(a)
        if include_H:
#            objective = cp.Minimize(sum_S - cp.log(sum_D) + 2*np.diag(M_W)@a)
#            objective = cp.Minimize(2 * lambd *sum_S + 2 * lambd* sum_W_S - cp.log(sum_D) - cp.log(sum_W_D) +  center_sum + + gamma*cp.pnorm(a, p=2)**2)
            objective = cp.Minimize(2 * lambd *sum_S + 2 * lambd* sum_W_S - cp.log(sum_D) - cp.log(sum_W_D) +  center_sum + gamma*cp.pnorm(a, p=2)**2)
        else:
            objective = cp.Minimize(sum_S - cp.log(sum_D))
        constraints = [0 <= a]
        prob = cp.Problem(objective, constraints)
        prob.solve(warm_start=True, verbose=False)
#        print(objective.value)
        
#        a0 = np.array([1,1,1,1,1,1,0,0,0,0,0,0])
#        sum_S0 = M_S @ a0
#        sum_D0 = M_D @ np.sqrt(a0)
#        sum_W_S0 = M_W_S @ a0
#        sum_W_D0 = M_W_D @ np.sqrt(a0)
#        ob0 = 2 * sum_S0 + 2 * sum_W_S0 - np.log(sum_D0) - np.log(sum_W_D0) + center_vec @ a0
#        print(ob0)
        
        print('lambd=%d, sum(abs(a))=%2.3f'% ( lambd, np.abs(a.value)@np.repeat(1,p)))
        print('Diagonal entries of the proposed method')
        ans = np.diag(a.value/np.linalg.norm(a.value))
#        ans = np.diag(a.value)
        print(np.diag(ans))
        return ans
    else:
        # TODO full-rank A case
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

#def metric_learn_A(X:'N by p data matrix', S:'similar set', D:'dissimilar set', H:'infered matrix from step 1'):
#    N, p = X.shape
#    M_S = np.zeros((p,p))
#    for i, j in S:
#        M_S += np.outer(X[i,:]-X[j,:], X[i,:]-X[j,:])
#    M_D = np.zeros((p,p))
#    for i, j in D:
#        M_D += np.outer(X[i,:]-X[j,:], X[i,:]-X[j,:])
#    W = np.dot(H, H.T)
#    L = np.diag(np.sum(W, axis=1))
#    M_W = np.dot(np.dot(X.T, L - W), X)
#    with Model("metric_learn") as M:
##        mimic the process of 11.6.3 of https://docs.mosek.com/9.0/pythonfusion.pdf
##        M.setLogHandler(sys.stdout)
##        Y = M.variable(Domain.inPSDCone(2*p))
##        A = Y.slice([0, 0], [p, p])
##        Z = Y.slice([0, p], [p, 2 * p])
##        DZ = Y.slice([p, p], [2 * p, 2 * p])
##        # Z is lower-triangular
##        M.constraint(Z.pick([[i,j] for i in range(n) for j in range(i+1,p)]), Domain.equalsTo(0.0))
##        # DZ = Diag(Z)
##        M.constraint(Expr.sub(DZ, Expr.mulElm(Z, Matrix.eye(p))), Domain.equalsTo(0.0))        
##        prod_greater_0(M, DZ.diag(), 1)
##        t2 = M.variable("t2", 1, Domain.greaterThan(1.0))
##        geometric_mean(M, DZ.diag(), t2) # 1 <= det(A)^(1/p) i.e. logdet(X) >= 0   
##        
#        ## trace + l2 constraints
##        Y = M.variable(Domain.inPSDCone(2*p))
##        A = Y.slice([0, 0], [p, p])
##        Z = Y.slice([0, p], [p, 2 * p])
##        DZ = Y.slice([p, p], [2 * p, 2 * p])
##        # Z is lower-triangular
##        M.constraint(Z.pick([[i,j] for i in range(n) for j in range(i+1,p)]), Domain.equalsTo(0.0))
##        # DZ = Diag(Z)
##        M.constraint(Expr.sub(DZ, Expr.mulElm(Z, Matrix.eye(p))), Domain.equalsTo(0.0))        
####        prod_greater_0(M, DZ.diag(), 1)
####        t2 = M.variable("t2", 1, Domain.greaterThan(1.0))
###        
###        M.constraint(DZ.diag().index(0), Domain.lessThan(1.0))
###        M.constraint(Expr.mul(DZ.diag(), np.append(0,np.repeat(1,p-1)).reshape(-1,1)), Domain.greaterThan(1.0))
#        ##
##        ## Eric Xing's
##        A = M.variable(Domain.inPSDCone(p))
##        M.objective(ObjectiveSense.Minimize, (Expr.add(Expr.dot(M_W, A), Expr.dot(M_S, A))))
###        M.objective(ObjectiveSense.Minimize, (Expr.dot(M_S, A)))
##        M.constraint( Expr.dot(M_D, A), Domain.greaterThan(1.0))
##        ##
#       
#        ## reverse Eric Xing's to find the most noisy dimension
#        A = M.variable(Domain.inPSDCone(p))
#        M.objective(ObjectiveSense.Maximize, (Expr.add(Expr.dot(M_W, A), Expr.dot(M_S, A))))
##        M.objective(ObjectiveSense.Maximize,  Expr.dot(M_S, A))
#        M.constraint( Expr.dot(M_D, A), Domain.lessThan(1.0))
#        ##
#        ## Eric Xing + trace
##        Y = M.variable(Domain.inPSDCone(2*p))
##        A = Y.slice([0, 0], [p, p])
##        Z = Y.slice([0, p], [p, 2 * p])
##        DZ = Y.slice([p, p], [2 * p, 2 * p])
##        # Z is lower-triangular
##        M.constraint(Z.pick([[i,j] for i in range(n) for j in range(i+1,p)]), Domain.equalsTo(0.0))
##        # DZ = Diag(Z)
##        M.constraint(Expr.sub(DZ, Expr.mulElm(Z, Matrix.eye(p))), Domain.equalsTo(0.0))   
##        M.constraint(Expr.mul(DZ.diag(), np.append(1,np.repeat(1,p-1)).reshape(-1,1)), Domain.greaterThan(1.0))
###        M.constraint(DZ.diag().index(0), Domain.equalsTo(1.0))
##        M.objective(ObjectiveSense.Minimize, (Expr.add(Expr.dot(M_W, A), Expr.dot(M_S, A))))
##        M.constraint( Expr.dot(M_D, A), Domain.greaterThan(1.0))
#        
###        prod_greater_0(M, DZ.diag(), 1)
###        t2 = M.variable("t2", 1, Domain.greaterThan(1.0))
##        
##        M.constraint(DZ.diag().index(0), Domain.lessThan(1.0))
##        M.constraint(Expr.mul(DZ.diag(), np.append(0,np.repeat(1,p-1)).reshape(-1,1)), Domain.greaterThan(1.0))
#        
##        M.objective(ObjectiveSense.Minimize, Expr.sub(Expr.add(Expr.dot(1/N/N*M_W, A), Expr.dot(1/len(S)*M_S, A)), Expr.dot(1/len(D)*M_D, A)))
##        M.objective(ObjectiveSense.Minimize,  Expr.add(Expr.dot(M_W, A), Expr.dot(M_S, A)))
##        M.constraint(Expr.dot(1/len(D)*M_D, A), Domain.greaterThan(1.0))
##        M.constraint(Expr.dot(M_D, A), Domain.greaterThan(1.0))
##        M.objective(ObjectiveSense.Minimize, Expr.add(Expr.add(Expr.dot(M_W, A), Expr.dot(M_S, A)), (Expr.dot(M_D, A))))
##        M.objective(ObjectiveSense.Minimize, Expr.add(Expr.dot(1/N/N*M_W, A), Expr.dot(1/len(S)*M_S, A)))
##        t = M.variable("t", 1, Domain.greaterThan(0.0))
##        M.objective(ObjectiveSense.Minimize, Expr.add(t, Expr.dot(1/len(S)*M_S, A)))
##        M.objective(ObjectiveSense.Minimize, Expr.sub(Expr.add(t, Expr.dot(1/len(S)*M_S, A)), Expr.dot(1/len(D)*M_D, A)))
##        M.constraint("slack", Expr.sub(Expr.dot(1/N/N*M_W, A),t), Domain.lessThan(0.0))
#        M.solve()
#        return np.array(A.level()).reshape((p,p))

#def metric_learn_A(X:'N by p data matrix', S:'similar set', D:'dissimilar set', H:'infered matrix from step 1'):
#    N, p = X.shape
#    M_S = np.zeros((p,p))
#    for i, j in S:
#        M_S += np.outer(X[i,:]-X[j,:], X[i,:]-X[j,:])
#    M_D = np.zeros((p,p))
#    for i, j in D:
#        M_D += np.outer(X[i,:]-X[j,:], X[i,:]-X[j,:])
#    W = np.dot(H, H.T)
#    L = np.diag(np.sum(W, axis=1))
#    M_W = np.dot(np.dot(X.T, L - W), X)
#    with Model("metric_learn") as M:
##        mimic the process of 11.6.3 of https://docs.mosek.com/9.0/pythonfusion.pdf
#        A = M.variable(Domain.inPSDCone(p))
#        M.objective(ObjectiveSense.Minimize, Expr.add(Expr.dot(1/N/N*M_W, A), Expr.dot(1/len(S)*M_S, A)))
#        M.constraint(Expr.dot(1/len(D)*M_D, A), Domain.greaterThan(1.0))
##        M.objective(ObjectiveSense.Minimize, Expr.add(Expr.add(Expr.dot(M_W, A), Expr.dot(M_S, A)), (Expr.dot(-M_D, A))))
##        M.objective(ObjectiveSense.Minimize, Expr.add(Expr.dot(1/N/N*M_W, A), Expr.dot(1/len(S)*M_S, A)))
##        t = M.variable("t", 1, Domain.greaterThan(0.0))
##        M.objective(ObjectiveSense.Minimize, Expr.add(Expr.add(t, Expr.dot(1/len(S)*M_S, A)), Expr.dot(-1/len(D)*M_D, A)))
##        M.constraint("slack", Expr.sub(Expr.dot(1/N/N*M_W, A),t), Domain.lessThan(0.0))
#        M.solve()
#        return np.array(A.level()).reshape((p,p))

#def metric_learn_A(X:'N by p data matrix', S:'similar set', D:'dissimilar set', H:'infered matrix from step 1'):
#    N, p = X.shape
#    M_S = np.zeros((p,p))
#    for i, j in S:
#        M_S += np.outer(X[i,:]-X[j,:], X[i,:]-X[j,:])
#    M_D = np.zeros((p,p))
#    for i, j in D:
#        M_D += np.outer(X[i,:]-X[j,:], X[i,:]-X[j,:])
#    W = np.dot(H, H.T)
#    L = np.diag(np.sum(W, axis=1))
#    M_W = np.dot(np.dot(X.T, L - W), X)      
#    eps = 1
#    gamma = 0.1
##    ans = eps * np.linalg.inv(1/N/N*M_W + 1/len(S)*M_S - 1/len(D)*M_D + gamma*np.eye(p))
#    ans = eps * np.linalg.inv(M_W + M_S - M_D + gamma*np.eye(p))
#    return ans
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
#    P1 = 7# true space dimension
#    P2 = 3 # noisy space dimension
#    X0, y, K, num_per_class, scale = load_high_dim3(P1,P2,10,seed=1)
#    X = X0
#    p = X.shape[1]
#    N = X.shape[0]
#    U = set(combinations(range(N),2))
#    S = set()
#    D = set()
#    num_pair = int(N*(N-1)/2) # number of constraints
#    while len(S) == 0 or len(D) == 0:
#        U = set(combinations(range(N),2))
#        S = set()
#        D = set()
#        S, D, U = add_query(y, S, D, U, num_pair)
#    H = np.zeros((N, K))
#    for n in range(N):
#        H[n, y[n]] = 1
##    H = np.random.randn(N*K).reshape(N,K)
#    fig, ax = plot3D(X, y)
#    ax.set_title('original space')
#    final_P = np.eye(p)
#    for d in range(P2):
#        A = metric_learn_A(X, S, D, H)
#        P, w,_ = np.linalg.svd(A)
##        print(w)
##    plot3D(X, y)
##    plot3D(transform(X, A), y)
#    # P is the projection matrix given the eigen values are [1, 0, 0, ...]
#        fig, ax = plot3D(transform(X, A), y)
#        ax.set_title('projected space')
#        P = proj_mat(A)
#        X = np.dot(X, np.eye(p) - P)
#        final_P = final_P.dot(np.eye(p) - P)
#        fig, ax = plot3D(X, y)
#        ax.set_title('orthogonal space')
#        print(np.linalg.svd(final_P)[1])   
#    plot3D(X0.dot(final_P), y)
#    plot3D(X0[:,3:],y)
#    plt.show()


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
#            P = metric_learn_backward(P2, X, S, D, H)
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
    # plot
    
#    np.savez('step2_ARI_true_H', result_ARI_Kmeans=result_ARI_Kmeans, reuslt_ARI_PCKmeans=result_ARI_PCKmeans, reuslt_ARI_MPCKmeans=result_ARI_MPCKmeans, result_ARI=result_ARI)    
    fig, ax = plt.subplots()
    ax.boxplot(result_ARI)
    ax.set_xticklabels(num_constraints)
    ax.set_xlabel('number of constraints')
    ax.set_ylabel('ARI')
#    fig.savefig('step2_ARI_true_H.png')    
#        plot3D(X[:,:3],y)
    
    fig, ax = plt.subplots()
    ax.boxplot(result_ARI_Kmeans)
    ax.set_xticklabels(num_constraints)
    ax.set_xlabel('number of constraints')
    ax.set_ylabel('ARI_Kmeans')
#    fig.savefig('step2_ARI_Kmeans_true_H.png') 
    
    fig, ax = plt.subplots()
    ax.boxplot(result_ARI_PCKmeans)
    ax.set_xticklabels(num_constraints)
    ax.set_xlabel('number of constraints')
    ax.set_ylabel('ARI_PCKmeans')
#    fig.savefig('step2_ARI_PCKmeans_true_H.png') 
    
#    fig, ax = plt.subplots()
#    ax.boxplot(result_ARI_MPCKmeans)
#    ax.set_xticklabels(num_constraints)
#    ax.set_xlabel('number of constraints')
#    ax.set_ylabel('ARI_MPCKmeans')
#    fig.savefig('step2_ARI_MPCKmeans_true_H.png') 
        