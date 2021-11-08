# -*- coding: utf-8 -*-
"""
Helper functions

@author: Yujia Deng
"""
import numpy as np
from math import *
from itertools import combinations
from random import sample
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from active_semi_clustering.semi_supervised.pairwise_constraints import PCKMeans, MPCKMeans, MPCKMeansMF, COPKMeans
import metric_learn
import os
from sklearn.datasets import make_circles, make_moons, load_wine, load_digits, load_iris, load_breast_cancer, fetch_olivetti_faces, fetch_covtype
from active_semi_clustering.exceptions import EmptyClustersException
from active_semi_clustering.active.pairwise_constraints import ExampleOracle, MinMax, NPU, ExploreConsolidate

def generate_points_on_sphere(r, K, p):
    """
    Generate K points uniformly on a p-dimensional sphere with radius r.
    p >= 2
    return: list of list
    """
    def angle_to_coord(angles):
        x = []
        cum = 1 # record cumulitive product of sines
        for p in range(len(angles)):
            x += [r * cum * cos(angles[p])]
            cum  *= sin(angles[p])
        x.append( r * cum)
        return x
    
    return [angle_to_coord([i/K*pi] * (p-2) + [i/K*2*pi]) for i in range(K)]

def load_sphere(P, K, n_per_cluster, r, sigma=1, seed=1,random_scale=False):
    """
    Generate data with K clusters and n_per_cluster number of data points per cluster. The data is from Gaussian mixture model with K mixtures and the centers are uniformly located on a P dimensional sphere with radius r.
    """
    rng = np.random.RandomState(seed)
    X = np.zeros((1,P))
    y = np.zeros((1,))
    A = np.zeros((P,P))
    centers = generate_points_on_sphere(r, K, P)
    for k in range(K):
        mu0 = centers[k]
        Sigma = sigma**2*np.diag(np.repeat(1, P))
        X0 = rng.multivariate_normal(mu0, Sigma, n_per_cluster)
        y0 = np.repeat(k, n_per_cluster)
        X = np.row_stack((X, X0))
        y = np.concatenate((y, y0))
    X = X[1:,:]
    y = y[1:]
    if random_scale:
        scale = np.square(rng.normal(1,1,P))
    else:
        scale = np.ones(P)
    X = transform(X, np.diag(scale))
    return X, y, scale

def plot2D(X, y):
    """
    Plot a 2D scatter plot showing the first two dimesnions of X colored by label y.
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.scatter(X[:, 0], X[:, 1], c=y)
    return fig, ax

    
def load_moon(n, p_noise, mu_noise, sd_gaussian, seed=1):
    """
    Generate moon shape clusters (2 clusters) with additional irrelevant features
    n: sample size
    p_noise: dimensions of the additional irrelevant efatures
    mu_noise: strength of the additional irrelevant features
    ad_guassian: noisy level of the moon shape clusters (in the first 2 dimensions)
    """
    X0, y = make_moons(n_samples=n, noise=sd_gaussian, random_state=seed)
    rng = np.random.RandomState(seed)
    X1, _, _, _ = load_opposite(p_noise, n, mu_noise, seed+1)
    
    X1 = rng.permutation(X1)
    X = np.column_stack((X0, X1[:X0.shape[0],:]))
    return X, y

def load_circles(n, p_noise, mu_noise, factor, circle_noise, seed=1):
    """
    Generate two circle clusters (K=2) with with additional irrelevant features
    n: sample size
    p_noise: dimensions of the additional irrelevant features
    mu_noise: strength of the additional irrelevant features
    factor: diameter of the inner circle
    circle_noise: noisy level of the moon shape clusters (in the first 2 dimensions)
    """
    X0, y = make_circles(n, factor=factor, noise=circle_noise, random_state=seed)
    rng = np.random.RandomState(seed)
    X1, _, _, _ = load_opposite(p_noise, n, mu_noise, seed+1)
    
    X1 = rng.permutation(X1)
    X = np.column_stack((X0, X1[:X0.shape[0],:]))
    return X, y
    
    

def load_simulation_sphere(K, P1, P2,  n_per_cluster, r=1, sigma=1, seed=1, random_scale=False):   
    """
    Generate data with K clusters. The cluster label is determined by the first P1 dimensions such that x=(x1, x2), where x1 is from Gaussian mixture model with K mixtures and the centers are uniformly located on a P1 dimensional sphere with radius r; x2 is from Gaussian mixture model with P2 clusters and each cluster center is at (..,0,mu,0,..).
    """
    rng = np.random.RandomState(seed)
    X0, y0, scale_0 = load_sphere(P1, K, n_per_cluster, r, sigma, seed, random_scale)
    X1, _, _, scale_1 = load_high_dim2(P2, ceil(K*n_per_cluster/P2), r, seed+1, random_scale)
    X1 = rng.permutation(X1)
    X = np.column_stack((X0, X1[:X0.shape[0], :]))
    y = y0
    scale = np.concatenate((scale_0, scale_1))
    return X, y, K, n_per_cluster, scale
    
def reader(path):
    f = open(path, 'r')
    content = f.read()

    data = []
    buffer = ''
    for c in content:
        if c == '(':
            buffer = ''
        elif c == ')':
            line = [str(num) for num in buffer.split()]
            data.append(line)
        else:
            buffer += c
    return data

def load_data(data_name): 
    if data_name == 'MEU-Mobile':
        data = pd.read_excel('./datasets/MEU-Mobile KSD 2016.xlsx', header=0)
        y = np.array(data['Subject'], dtype='int')
        X = np.array(data.iloc[:, 1:], dtype='float')
        # y may contain negative values corresponding to blank lines
        idx = np.where((y<10) & (y>=0))[0]
        X = X[idx, :]
        y = y[idx]
        
    elif data_name == 'urban_land_cover':
        data1 = pd.read_csv('./datasets/Urban land cover/training.csv', header=0)
        data2 = pd.read_csv('./datasets/Urban land cover/testing.csv', header=0)
        data = pd.concat([data1, data2])
        X = np.array(data.iloc[:, 1:])
        y = pd.CategoricalIndex(data['class']).codes
   
    elif data_name == 'breast_cancer':
        X, y = load_breast_cancer(return_X_y=True)

    else:
        print('No such dataset')
    return X, y

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

def plot3D(X,Y):
    x_min, x_max = X[:, 0].min() * .9, X[:, 0].max() * 1.1
    y_min, y_max = X[:, 1].min() * .9, X[:, 1].max() * 1.1
    z_min, z_max = X[:, 2].min() * .9, X[:, 2].max() * 1.1
    
    max_range = np.max((x_max-x_min, y_max-y_min, z_max-z_min))/2
    mid_x = 0.5*(x_min + x_max)
    mid_y = 0.5*(y_min + y_max)
    mid_z = 0.5*(z_min + z_max)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:,0],X[:,1],X[:,2],c=Y)
    ax.set_xlim(mid_x-max_range, mid_x+max_range)
    ax.set_ylim(mid_y-max_range, mid_y+max_range)    
    ax.set_zlim(mid_z-max_range, mid_z+max_range)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_aspect('equal')
    ax.view_init(30, 45)
    return fig, ax

def mat_normalize(A):
    w, v = np.linalg.eig(A)
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


def load_mix(P1, P2, N, mu=5, seed=1, rho:'mixing rate'=0.5):
    X, y, num_class, num_per_class, scale = load_high_dim4(P1, P2, N, mu, seed, random_scale=False, rotate=False)
    # mix the true feature and the noisy features
    tmp = np.random.randn(P1+P2, P1+P2)
    A = tmp.dot(tmp.T)
    X = transform(X, A)
    return X, y, num_class, num_per_class, scale, A
    
def load_rank1(P1, P2, K, N, mu=3, seed=1):
    rng = np.random.RandomState(seed)
    mu0 = mu * np.repeat(np.arange(K), N)
    X = [rng.normal(mu0[i]) for i in range(K*N)]
    X0 = np.array(X).reshape(K*N, 1)
    X1,_,_,_ = load_opposite(P2, int(N*K/P2), mu, seed)
    y = np.repeat(np.arange(K), N)
    if X0.shape[0] != X1.shape[0]:
        n_total = min(X0.shape[0], X1.shape[0])
        X0 = X0[:n_total, :]
        X1 = X1[:n_total, :]
        y = y[:n_total]
    
    X = np.concatenate((X0, X1), 1)
    n_total = X.shape[0]
    return X, y, K, n_total
   
def load_rank2(P1, P2, K, N, mu=3, seed=1):
    rng = np.random.RandomState(seed)
    X0 = []
    for k in range(K):
        mu0 = (mu * cos(2*pi/K*k), mu * sin(2*pi/K*k))
        for n in range(N):
            X0.append(rng.multivariate_normal(mu0, np.eye(2)))
    X0 = np.array(X0)
    X1,_,_,_ = load_opposite(P2, int(N*K/P2), mu, seed)
    y = np.repeat(np.arange(K), N)
    if X0.shape[0] != X1.shape[0]:
        n_total = min(X0.shape[0], X1.shape[0])
        X0 = X0[:n_total, :]
        X1 = X1[:n_total, :]
        y = y[:n_total]
    
    X = np.concatenate((X0, X1), 1)
    n_total = X.shape[0]
    return X, y, K, n_total
    
def load_rankr(P1, P2, K, N, mu=3, seed=1):
    """
    P1 is the dimension of the true features (i.e. r).
    P2 is the dimension of the noisy features.
    K is the number of clusters.
    """
    rng = np.random.RandomState(seed)
    X0 = []


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