# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 13:19:58 2020

@author: Yujia Deng
"""
from cobra.src.cobra import *
from cobras_ts.cobras_kmeans import COBRAS_kmeans
from cobras_ts.querier.labelquerier import LabelQuerier
from Step1_Impute import infer_membership_from_label
from helper import *
from active_random_MPCKmeans_parallel import ARI_semi_active_with_constraints, ARI_semi_active
from active_semi_clustering.exceptions import EmptyClustersException
from active_semi_clustering.active.pairwise_constraints import ExampleOracle

from proposed_clusterer import proposed_clusterer
from mynpu_metric import *
from csp import CSP

def ARI_nc_COBRA(X, y, K, n_super_instance):
    clusterer = COBRA(n_super_instance)
    clusterings, _, ml, cl = clusterer.cluster(X, y, range(len(y)))
    return adjusted_rand_score(y, clusterings[-1]), len(ml) + len(cl)

def ARI_nc_COBRAS(X, y, budget):
    clusterer = COBRAS_kmeans(X, LabelQuerier(y), budget)
    clustering, intermediate_clusterings, runtimes, ml, cl = clusterer.cluster()
    return adjusted_rand_score(y, clustering.construct_cluster_labeling()), len(ml) + len(cl)


def repeat_COBRA(n_super_instance, rep, P1, P2, mu, A0):
    print("rep=%d, n_super_instance=%d" % (rep, n_super_instance))
    X0, y, K, num_per_class, scale = load_high_dim4(P1, P2, N=15, mu=mu, seed=rep)
    X = transform(X0, A0)
    N, p = X.shape
    ARI, n_query = ARI_nc_COBRA(X, y, K, n_super_instance)
    print('n_query=%d, ARI_COBRA=%2.3f' %(n_query, ARI))
    return (n_query, ARI)


def ARI_active_old(X, y, K, max_nc, metric_learn_method='mpckmeans', impute_method='default', weighted=False, uncertainty='random_forest',diag=True, true_H=False, include_H=True, lambd=1, gamma=100, rank=1, num_p=0, verbose=None, penalized=False, initial='default', request_nc=None):
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
    elif metric_learn_method.lower() == 'proposed_csp':
        clusterer = proposed_clusterer(n_clusters=K, clusterer='csp')
    elif metric_learn_method.lower() == 'csp':
        clusterer = CSP(n_clusters=K)

    active_learner = NPU_old(clusterer, impute_method=impute_method, weighted=weighted, uncertainty=uncertainty, initial=initial, penalized=penalized, lambd=lambd, gamma=gamma, num_p=num_p, diag=diag, true_H=true_H) # use old NPU
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


def ARI_active(X, y, K, max_nc, metric_learn_method='mpckmeans', impute_method='default', weighted=False, uncertainty='random_forest',diag=True, true_H=False, include_H=True, lambd=1, gamma=100, rank=1, num_p=0, verbose=None, penalized=False, initial='default', request_nc=None):
    print(metric_learn_method.lower())
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
    elif metric_learn_method.lower() == 'proposed_csp':
        clusterer = proposed_clusterer(n_clusters=K, clusterer='csp')
    elif metric_learn_method.lower() == 'csp':
        clusterer = CSP(n_clusters=K)

    active_learner = NPU(clusterer, impute_method=impute_method, weighted=weighted, uncertainty=uncertainty, initial=initial, penalized=penalized, lambd=lambd, gamma=gamma, num_p=num_p, diag=diag, true_H=true_H)
    active_learner.get_true_label(y)
    active_learner.fit(X, oracle, request_nc=request_nc)
    print('hist_nc')
    print(active_learner.hist_nc)

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
    