"""

"""

from sequential_output_npu import *
import sys
import os
import errno
from Step1_Impute import infer_membership_from_label
from active_semi_clustering.exceptions import EmptyClustersException
from active_semi_clustering.active.pairwise_constraints import ExampleOracle

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

# result_MPCKmeans, _, _, _ = ARI_active(metric_learn_method='mpckmeans', X=X, y=y, K=K, max_nc=max_nc, impute_method='default', weighted=False, uncertainty='random_forest',diag=True, include_H=True, lambd=0, gamma=0, rank=1, num_p=0, verbose=None, penalized=False, initial='default', request_nc=request_nc)
# result_MPCKmeans = dict()

result_proposed_no_penalty, result_proposed_penalty, A_hist, A_hist_penalize = ARI_active(metric_learn_method='proposed', X=X, y=y, K=K, max_nc=max_nc, impute_method='default', weighted=False, uncertainty='random_forest',diag=True, true_H=False, include_H=True, lambd=1000, gamma=100, rank=1, num_p=int(P2/2), verbose=None, penalized=True, initial='default', request_nc=request_nc)

###########
# For COBRA
###########
# n_super_instances = range(10, 200, 20)
# result_COBRA = dict()
# for nsi in n_super_instances:
#     ARI, n_query = ARI_nc_COBRA(X, y, K, nsi)
#     result_COBRA[nsi] = (n_query, ARI)

# np.savez(os.path.join(save_path, ('rep%d.npz' % rep)), result_MPCKmeans=result_MPCKmeans, result_proposed_no_penalty=result_proposed_no_penalty, result_proposed_penalty=result_proposed_penalty, result_COBRA=result_COBRA, A_hist=A_hist, A_hist_penalize=A_hist_penalize)


# print(result_MPCKmeans)
# print(result_proposed_no_penalty)
print(result_proposed_penalty)
# print(result_COBRA)
