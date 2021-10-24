import numpy as np
import scipy as sp
from numpy.linalg import *
from scipy import spatial
from sklearn.cluster import KMeans
from active_semi_clustering.exceptions import EmptyClustersException
class CSP:
    """
    Python version of constrained spectral clustering, forked from https://github.com/peisuke/ConstrainedSpectralClustering/blob/master/constrained_spectral_clustering_K.py,    
    adapted from https://github.com/gnaixgnaw/CSP/blob/master/csp_K.m
    """
    def __init__(self, n_clusters=3):
        self.K = self.n_clusters = n_clusters
    
    def nnz(self, x):
        """
        Number of nonzero matrix elements
        """
        return sum(x != 0)
    def create_affinity_matrix(self, X, K_nn=16):
        """
        use K-nearest neighbor to construct the affinity matrix. TODO: other kernel methods.
        """
        tree = spatial.KDTree(X)
        dist, idx = tree.query(X, k=K_nn)
        
        idx = idx[:,1:]
        
        nb_data, _ = X.shape
        A = np.zeros((nb_data, nb_data))
        for i, j in zip(np.arange(nb_data), idx):
            A[i, j] = 1
        A = np.maximum(A.T, A)

        return A
        
    def fit(self, X, ml=[], cl=[]):
        K = self.K
        N, p = X.shape
        A = self.create_affinity_matrix(X)
        # constraints matrix
        Q = np.eye(N)
        for i, j in ml:
            Q[i, j] = Q[j, i] = 1
        for i, j in cl:
            Q[i, j] = Q[j, i] = -1
        D = np.diag(np.sum(A, axis=1))
        vol = np.sum(A)

        D_norm = np.linalg.inv(np.nan_to_num(np.sqrt(D)))
        L_norm = np.eye(*A.shape) - D_norm.dot(A.dot(D_norm))
        Q_norm = D_norm.dot(Q.dot(D_norm))

        L = L_norm
        Q = Q_norm
        #######
        
        # alpha < K-th eigenval of Q_norm
        alpha = 0.6 * sp.linalg.svdvals(Q_norm)[K]
        Q1 = Q_norm - alpha * np.eye(*Q_norm.shape)
        
        val, vec = sp.linalg.eig(L_norm, Q1)
        
        vec = vec[:,val >= 0]
        vec_norm = (vec / np.linalg.norm(vec, axis=0)) * np.sqrt(vol)

        costs = np.multiply(vec_norm.T.dot(L_norm), vec_norm.T).sum(axis=1)
        ids = np.where(costs > 1e-10)[0]
        min_idx = np.argsort(costs[ids])[0:K]
        min_v = vec_norm[:,ids[min_idx]]

        U = D_norm.dot(min_v)

        #####
        # _, lam, _ = svd(Q)
        # # beta = 0.6 * lam[K]
        # beta = (lam[K] + lam[K-1]) / 2 - 1e-6

        # Q1 = Q - beta * np.eye(N)
        # val, vec = sp.linalg.eig(L_norm, Q1)
        # # w, vec = eig(inv(Q1).dot(L))

        # for i in range(N):
        #     vec[:, i] = vec[:, i] / norm(vec[:, i])

        # # find feasible cuts
        # satisf = np.diag(vec.T.dot(Q1).dot(vec))
        # I_ = np.where(satisf >= 0)[0]
        
        # cost = np.diag(vec[:, I_].T.dot(L).dot(vec[:, I_]))
        # ind = sorted(range(len(cost)), key=lambda i: cost[i])

        # # remove trivial cuts
        # i = 0
        # while True:
        #     if self.nnz(vec[:, I_[ind[i]]] > 0) != 0 and self.nnz(vec[:, I_[ind[i]]] < 0) != 0 :
        #         break
        #     i += 1
        # ind = ind[i:]


        # # output cluster indicators
        # ind = ind[:min(len(ind), K-1)]
        # cost = cost[ind]
        # U = vec[:, I_[ind]]
        # for i in range(U.shape[1]):
        #     U[:, i] = D_norm.dot(U[:, i]).dot(np.sqrt(vol))
        # U = np.real(U)
        ####
        U = np.real(U)
        while True:
            try:
                model = KMeans(n_clusters=K).fit(U)
                break
            except EmptyClustersException:
                print('Empty cluster')
                continue
        
        self.labels_ = model.labels_
        return self
        

        
