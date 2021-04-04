
import numpy as np
from math import *
from itertools import combinations
from random import sample
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from active_semi_clustering.semi_supervised.pairwise_constraints import PCKMeans, MPCKMeans, MPCKMeansMF, COPKMeans
import metric_learn
import os
from sklearn.datasets import load_wine, load_digits, load_iris, load_breast_cancer, fetch_olivetti_faces, fetch_covtype
from active_semi_clustering.exceptions import EmptyClustersException
from active_semi_clustering.active.pairwise_constraints import ExampleOracle, MinMax, NPU, ExploreConsolidate

def mat_target(P1, A):
    """
    return the target matrix wrt the transformation matrix A0
    """
    tmp = np.repeat(1, A.shape[0])
    tmp[P1:] = 0
    return mat_sqrt_inv(A).dot(np.diag(tmp)).dot(mat_sqrt_inv(A))

def output(file, result, setting_name):
    file.write('\n'+setting_name)
    res_mean = np.mean(result, 0)
    res_sd = np.std(result, 0)
    for n in range(len(res_mean)):
        file.write('\t %2.3f(%2.3f)' % (res_mean[n], res_sd[n]))


def metric_learn_SD(X, y, S, D, diagonal=True, verbose=True, A0=None):
    mmc_r = metric_learn.MMC(diagonal=diagonal, verbose=verbose, A0=A0)
    a, b = zip(*S)
    c, d = zip(*D)
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    d = np.array(d)
    mmc_r.fit(X, constraints=(a,b,c,d))
    X_r = transform(X, mmc_r.A_)
    return X_r, mmc_r.A_

def ARI_clustering(X, y, C, method='kmeans', S=None, D=None):
    if method.lower() == 'kmeans':
        model = KMeans(n_clusters=C)
        model.fit(X)
    elif method.lower() == 'pckmeans':
        model = PCKMeans(n_clusters=C)
        model.fit(X, ml=list(S), cl=list(D))
    elif method.lower() == 'mpckmeansmf':
        # multiple full rank metric
        model = MPCKMeansMF(n_clusters=C)
        model.fit(X, ml=list(S), cl=list(D))    
    elif method.lower() == 'mpckmeans':
        model = MPCKMeans(n_clusters=C)
        model.fit(X, ml=list(S), cl=list(D))      
#        print('Diagonal entries from MPCKmeans')
#        print(np.diag(model.A))
    return adjusted_rand_score(y, model.labels_)

def add_query(y, S, D, U, n_pair, query_rule='random', pairs=None):
    if query_rule == 'random':    
        pairs = sample(U, n_pair)
    for (i,j) in pairs:
        if y[i] == y[j]:
            S = S.union({(i,j)})
        else:
            D = D.union({(i,j)}) 
    U = U - S - D
    return S, D, U

def mat_sqrt(A):
    """
    return V st A=V*V'
    """
    w, v = np.linalg.eigh(A)
    eps = 1e-8
    w = np.array([x if x>eps else 0 for x in w])
    assert np.all(w>=0), "A is not PSD"
    return np.dot(v, np.sqrt(np.diag(np.abs(w)))).dot(v.T)

def mat_sqrt_inv(A):
    """
    return V st A^-1=V*V
    mat_sqrt(A).dot(mat_sqrt_inv(A)) == I
    """
    w, v = np.linalg.eigh(A)
    eps = 1e-8
    w = np.array([x if x>eps else 0 for x in w])
    assert np.all(w>=0), "A is not PSD"
    return np.dot(v, np.sqrt(np.diag(1/np.abs(w)))).dot(v.T)

def proj_mat(A):
    U, d, _ = np.linalg.svd(A)
    p = np.argmax(d/d[0] < 1e-4) # the first index that d < 1e-4 
    if p == 0:
        p = len(d)
    M = U[:,:p]
    P = M.dot(np.linalg.inv(M.T.dot(M))).dot(M.T)
    return P

def transform(X, A):
    return np.dot(X, mat_sqrt(A))

def mat_normalize(A):
    w, v = np.linalg.eig(A)
#    w = [0 if x < 1e-4 for x in w]
    tmp = np.diag(1/np.sqrt(w))
    return(np.dot(np.dot(tmp, A),tmp))     
    


def load_sim(sigma):
    N = 25
    x0 = 15; y0 = -20; z0 = 0;
    x1 = 15; y1 = 20; z1 = 0;
    x2 = 20; y2 = 20; z2 = 0;
    x3 = 20; y3 = -20; z3 = 0;
    
    X0 = np.random.multivariate_normal((x0,y0,z0),np.diag((sigma**2,1,1)),N)
    X1 = np.random.multivariate_normal((x1,y1,z1),np.diag((sigma**2,1,1)),N)
    X2 = np.random.multivariate_normal((x2,y2,z2),np.diag((sigma**2,1,1)),N)
    X3 = np.random.multivariate_normal((x3,y3,z3),np.diag((sigma**2,1,1)),N)
    
    X = np.row_stack((X0, X1, X2, X3))
    y = np.concatenate((np.zeros((N*2,)), np.ones((N*2,)))).astype(int)
    return X, y

def load_high_dim(P, N):
    """
    P: number of dimensions 
    N: number of points per cluster
    """
    X = np.zeros((1,P))
    y = np.zeros((1,))
    inverse = np.zeros((P,P))
    num_class = 0
    for p in range(P):
        sigma = 1
        mu0 = np.zeros((P,))
        mu0[p] = 5
        mu1 = np.zeros((P,))
        mu1[p] = -5
        Sigma = sigma**2*np.diag(np.repeat(1, P))
        X0 = np.random.multivariate_normal(mu0, Sigma, N)
        X1 = np.random.multivariate_normal(mu1, Sigma, N)
        shrink = np.random.binomial(1, 0.5, 1)
        if shrink == 1: # closer
            X0 *= 1
            X1 *= 1
            y0 = np.concatenate((np.repeat(num_class, N), np.repeat(num_class+1, N)))#
            num_class += 2
            inverse[p,p] = 10
        else:
            X0 *= 1
            X1 *= 1
            y0 = np.repeat(num_class, 2*N).reshape((2*N,))
            num_class += 1
            inverse[p,p] = 0.1
        X = np.row_stack((X, X0, X1))
        y = np.concatenate((y, y0))
    X = X[1:,:]
    y = y[1:]
    return X, y, num_class, inverse

def load_high_dim2(P, N, mu=5, seed=1, random_scale=False):
    """
    Diagonal case with random scale. Generate datas with P clusters. Each cluster is centered at (..,0,mu,0,..) at the p-th dimension. 
    P: number of dimensions 
    N: number of points per cluster
    """
    rng = np.random.RandomState(seed)
    X = np.zeros((1,P))
    y = np.zeros((1,))
    A = np.zeros((P,P))
    inverse = np.zeros((P,P))
    num_class = 0
    for p in range(P):
        sigma = 1
        mu0 = np.zeros((P,))
        mu0[p] = mu
        Sigma = sigma**2*np.diag(np.repeat(1, P))
        X0 = rng.multivariate_normal(mu0, Sigma, N)
        y0 = np.repeat(p, N)
        num_class += 1
        X = np.row_stack((X, X0))
        y = np.concatenate((y, y0))
    
    X = X[1:,:]
    y = y[1:]
    if random_scale:
        scale = np.square(rng.normal(1,1,P))
    else:
        scale = np.ones(P)
    X = transform(X, np.diag(scale))
    return X, y, num_class, scale

def load_high_dim3(P1,P2,N, mu=5, seed=1,random_scale=False, rotate=False):
    """
    Generate X of size n*(P1+P2), whose label is based on the first P1 dimensions.
    """
    rng = np.random.RandomState(seed)
    a = np.lcm(P1, P2)
    N1 = int(a/P1*N)
    N2 = int(a/P2*N)
    num_class = P1
    num_per_class = N1
    X0, y0, foo, scale_0 = load_high_dim2(P1, N1, mu, seed, random_scale)
    X1, y1, foo, scale_1 = load_high_dim2(P2, N2, mu, seed+1, random_scale)
    X1 = rng.permutation(X1)
    X = np.column_stack((X0, X1))
    if rotate:
        print('random rotate')
        A = rng.randn(P1+P2, P1+P2)
        Q, R = np.linalg.qr(A)
        X = X.dot(Q)
    y =  np.repeat(range(P1), N1)
    scale = np.concatenate((scale_0, scale_1))
    return X, y, num_class, num_per_class, scale

def load_opposite(P, N, mu=10, seed=1, random_scale=False):
    rng = np.random.RandomState(seed)
    X = np.zeros((1,P))
    y = np.zeros((1,))
    num_class = 0
    for p in range(P):
        sigma = 1
        mu0 = np.zeros((P,))
        mu0[p] = -mu
        mu1 = np.zeros((P,))
        mu1[p] = mu
        Sigma = sigma**2*np.diag(np.repeat(1, P))
        if N%2 == 0:
            X0 = rng.multivariate_normal(mu0, Sigma, int(N/2))
            X1 = rng.multivariate_normal(mu1, Sigma, int(N/2))
        else:
            X0 = rng.multivariate_normal(mu0, Sigma, int(N/2))
            X1 = rng.multivariate_normal(mu1, Sigma, int(N/2) + 1)
        y0 = np.repeat(p, N)
        num_class += 1
        X = np.row_stack((X,np.row_stack((X1, X0))))
        y = np.concatenate((y, y0)) 
    X = X[1:,:]
    y = y[1:]
    if random_scale:
        scale = np.square(rng.normal(1,1,P))
    else:
        scale = np.ones(P)
    X = transform(X, np.diag(scale))
    return X, y, num_class, scale

def load_high_dim4(P1, P2, N, mu=5, seed=1,random_scale=False, rotate=False):
    """
    Generate X of size n*(P1+P2), whose label is based on the first P1 dimensions.
    The rest P2 dimensions are located in two clusters
    N: number of instances per cluster
    """
    rng = np.random.RandomState(seed)
    # N1 = int(P1*N)
    # a = np.lcm(P1, P2)
    # N1 = int(a/P1*N)
    # N2 = int(a/P2*N)
    N1 = N
    N2 = ceil(N1*P1/P2)
    num_class = P1
    num_per_class = N1
    X0, y0, foo, scale_0 = load_high_dim2(P1, N1, mu, seed, random_scale)
    X1, y1, foo, scale_1 = load_opposite(P2, N2, mu, seed+1, random_scale)
    
    X1 = rng.permutation(X1)
    X = np.column_stack((X0, X1[:X0.shape[0],:]))
    if rotate:
        print('random rotate')
        A = rng.randn(P1+P2, P1+P2)
        Q, R = np.linalg.qr(A)
        X = X.dot(Q)
    y =  np.repeat(range(P1), N1)
    scale = np.concatenate((scale_0, scale_1))
    return X, y, num_class, num_per_class, scale




def ARI_semi_active(X, y, K, nc, semi, active):
    oracle = ExampleOracle(y, max_queries_cnt=nc)
    if active.lower() == 'minmax':
        active_learner = MinMax(n_clusters=K)
        if semi.lower() == 'pckmeans':
            clusterer = PCKMeans(n_clusters = K)
        elif semi.lower() == 'mpckmeans':
            clusterer = MPCKMeans(n_clusters = K)
        elif semi.lower() == 'copkmeans':
            clusterer = COPKMeans(n_clusters = K)
            
    elif active.lower() == 'npu':      
        if semi.lower() == 'pckmeans':
            clusterer = PCKMeans(n_clusters = K)
        elif semi.lower() == 'mpckmeans':
            clusterer = MPCKMeans(n_clusters = K)
        elif semi.lower() == 'mpckmeansmf':
            clusterer = MPCKMeansMF(n_clusters = K)
        elif semi.lower() == 'copkmeans':
            clusterer = COPKMeans(n_clusters = K)
        active_learner = NPU(clusterer=clusterer)
        
        
    active_learner.fit(X, oracle)
    pairwise_constraints = active_learner.pairwise_constraints_
    clusterer.fit(X, ml=pairwise_constraints[0], cl=pairwise_constraints[1])
    return adjusted_rand_score(y, clusterer.labels_)    