
import sys
import os
import errno
from active_semi_clustering.exceptions import EmptyClustersException
from active_semi_clustering.active.pairwise_constraints import ExampleOracle
from helper import *

from proposed_clusterer import proposed_clusterer
from mynpu_metric import NPU

def infer_membership_from_label(S:'similar set', D:'dissimilar set', N:'number of samples', K:'number of clusters', lambd: 'multidirectional penalty weight'=1, rho:'multiplier weight'=10, eps=1e-3,inspect=False) -> "N by K array":
    nc = len(S) + len(D)
    inits = 5
    L_past = np.infty
    for init in range(inits): # try different initiations to avoid local extrema
        H = np.random.randn(N,K)
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
    if inspect:
        return ans
    else:
        return ans[0]

def ARI_active(X, y, K, max_nc, metric_learn_method='mpckmeans', impute_method='default', weighted=False, uncertainty='random_forest',diag=True, true_H=False, include_H=True, lambd=1, gamma=100, rank=1, num_p=0, verbose=None, penalized=False, initial='default', request_nc=None):
    if not request_nc:
        request_nc = max_nc
    oracle = ExampleOracle(y, max_queries_cnt = max_nc)
    if metric_learn_method.lower() == 'mpckmeans':
        clusterer = MPCKMeans(n_clusters=K)
    elif metric_learn_method.lower() == 'proposed':
        clusterer = proposed_clusterer(n_clusters=K)
    elif metric_learn_method.lower() == 'pckmeans':
        clusterer = PCKMeans(n_clusters=K)
    elif metric_learn_method.lower() == 'copkmeans':
        clusterer = COPKMeans(n_clusters=K)
        
    active_learner = NPU(clusterer, impute_method=impute_method, weighted=weighted, uncertainty=uncertainty, initial=initial, penalized=penalized, lambd=lambd, gamma=gamma, num_p=num_p, diag=diag, true_H=true_H)
    active_learner.get_true_label(y)
    active_learner.fit(X, oracle, request_nc=request_nc)
    
    result_no_penalty = dict()
    result_penalty = dict()
    A_hist = dict()
    A_hist_penalize = dict()
    for nc in request_nc:
        result_no_penalty[nc] = adjusted_rand_score(y, active_learner.hist_labels[nc])
        if len(active_learner.hist_A):
            A_hist[nc] = active_learner.hist_A[nc]
    if penalized:
        for nc in request_nc:
            result_penalty[nc] = adjusted_rand_score(y, active_learner.hist_labels_penalize[nc])     
            if len(active_learner.hist_A_penalize):
                A_hist_penalize[nc] = active_learner.hist_A_penalize[nc]
    return result_no_penalty, result_penalty, A_hist, A_hist_penalize



sys.argv = ['', 5, 30, 3, 60, 10, 0]
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
save_path = './simulation_high_dim_4/P1_%d_P2_%d_mu_%2.1f_N_per_cluster_%d_lambda_200/max_nc_%d/' % (P1, P2, mu, N_per_cluster, max_nc)
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

result_proposed_no_penalty, result_proposed_penalty, A_hist, A_hist_penalize = ARI_active(metric_learn_method='proposed', X=X, y=y, K=K, max_nc=max_nc, impute_method='default', weighted=False, uncertainty='random_forest',diag=True, true_H=False, include_H=True, lambd=1000, gamma=100, rank=1, num_p=int(P2/2), verbose=None, penalized=True, initial='default', request_nc=request_nc)


print(result_proposed_penalty)

