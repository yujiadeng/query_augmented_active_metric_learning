# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 20:28:06 2019

@author: Yujia Deng

Select the query based on the infered H, return 2K query
"""
from helper import *
#def inferred_query(H:'infered membership matrix', M:'transformation matrix', D:'dissimilar set', S:'similar set', U:'unlabeled set', X:'N x p data matrix', nc:'number of queries', K:'number or  clusters')->'length nc candidate set':
#    bu = 0
#    for i,j in S:
#        tmp = np.inner(np.inner(X[i,:]-X[j,:], M), np.inner(X[i,:]-X[j,:], M)) 
#        if tmp > bu:
#            bu = tmp
#    bl = np.infty
#    for i, j in D:
#        tmp = np.inner(np.inner(X[i,:]-X[j,:], M), np.inner(X[i,:]-X[j,:], M)) 
#        if tmp < bl:
#            bl = tmp
#    
#    recordu = dict()
#    recordl = dict()
#    for i, j in U:
#        tmp = np.inner(np.inner(X[i,:]-X[j,:], M), np.inner(X[i,:]-X[j,:], M))
#        tmp2 = np.inner(H[i,:], H[j,:])
#        if tmp2 > 1/K:
#            recordu[(i,j)] = tmp
#        if tmp2 < 1/K:
#            recordl[(i,j)] = tmp
#    # choose the nc/2 largest value from recordu
#    # choose the nc-Ku smallest value from recordl
#    
#    Ku = int(nc/2)
#    Kl = int(nc - Ku)
#    if len(recordu) <= Ku:
#        setu = set(recordu.keys())
#    else:
#        idx = np.argsort(-np.array(list(recordu.values())))
#        setu = [list(recordu.keys())[idx[k]] for k in range(Ku)]
#    
#    if len(recordl) <= Kl:
#        setl = set(recordl.keys())
#    else:
#        idx = np.argsort(np.array(list(recordl.values())))
#        setl = [list(recordl.keys())[idx[k]] for k in range(Kl)]
#    return set(setu) | set(setl)

def inferred_query(H:'infered membership matrix', M:'transformation matrix', D:'dissimilar set', S:'similar set', U:'unlabeled set', X:'N x p data matrix', nc:'number of queries', K:'number or  clusters')->'length nc candidate set':
    bu = 0
    for i,j in S:
        tmp = np.inner(np.inner(X[i,:]-X[j,:], M), np.inner(X[i,:]-X[j,:], M)) 
        if tmp > bu:
            bu = tmp
    bl = np.infty
    for i, j in D:
        tmp = np.inner(np.inner(X[i,:]-X[j,:], M), np.inner(X[i,:]-X[j,:], M)) 
        if tmp < bl:
            bl = tmp
    
    recordu = dict()
    recordl = dict()
    
    model = PCKMeans(n_clusters=K)
    model.fit(X.dot(M)) # projected X
    y_fit = model.labels_
    for i, j in U:
        
        tmp = np.inner(np.inner(X[i,:]-X[j,:], M), np.inner(X[i,:]-X[j,:], M))
#        tmp2 = np.inner(H[i,:], H[j,:])
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
    
        
if __name__ == '__main__':
    # run Step1_Impute.py before
    K = 6
    query_set = inferred_query(H, np.eye(9), D, S, U, X, 10, K)
