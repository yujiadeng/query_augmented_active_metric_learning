# -*- coding: utf-8 -*-
"""
Helper function to run MPCKmeans

@author: Yujia Deng
"""
from helper import *

def find_uncertain_pairs(y_fit, X, A, D, S, U, K, nc):
    recordu = dict()
    recordl = dict()
    for i, j in U:
        tmp = np.inner(np.inner(X[i,:]-X[j,:], A), X[i,:]-X[j,:])
        if y_fit[i] == y_fit[j]:
            recordu[(i,j)] = tmp
        else:
            recordl[(i,j)] = tmp
    # choose the nc/2 largest value from recordu
    # choose the nc-Ku smallest value from recordl 
    Ku = int(nc/2)
    Kl = int(nc - Ku)
    if len(recordu) <= Ku:
        setu = set(recordu.keys())
    else:
        idx = np.argsort(-np.array(list(recordu.values())))
        setu = [list(recordu.keys())[idx[k]] for k in range(Ku)]
    if len(recordl) <= Kl:
        setl = set(recordl.keys())
    else:
        idx = np.argsort(np.array(list(recordl.values())))
        setl = [list(recordl.keys())[idx[k]] for k in range(Kl)]
    return set(setu) | set(setl)

    
def active_MPCKmeans(X, y, K, B, B_per_step):
    N, p = X.shape
    U = set(combinations(range(N),2))
    S = set()
    D = set()
    for step in range(0, B, B_per_step):
        if step == 0:
            while len(S) < 3 or len(D) < 3:
                U = set(combinations(range(N),2))
                S = set()
                D = set()
                S, D, U = add_query(y, S, D, U, B_per_step)
        else:
            candidate_set = find_uncertain_pairs(y_fit, X, A, D, S, U, K, nc)
            S, D, U = add_query(y, S, D, U, nc, query_rule='fixed', pairs=candidate_set)
        nc = min(B_per_step, B-step)
        model = MPCKMeans(K)
        model.fit(X, ml=list(S), cl=list(D))
        A = model.A
        y_fit = model.labels_
    return A, y_fit
                
                
                
                
                
                
        