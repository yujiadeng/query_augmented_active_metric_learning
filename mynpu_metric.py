"""
Main class of the proposed active clustering tool

TODO: change the class name

@author: Yujia Deng
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from active_semi_clustering.active.pairwise_constraints.example_oracle import MaximumQueriesExceeded
from active_semi_clustering.exceptions import EmptyClustersException
import heapq
import time
class NPU:
    def __init__(self, clusterer=None, impute_method='default', weighted=False, uncertainty='random_forest', initial='default', penalized=False, lambd=0, gamma=0, num_p=0, n_tree=50, diag=True, true_H=False, **kwargs):
        """
        MEE method to query

        clusterer: char, method to do semi-supervised clustering, default is PCKmeans
        impute_method: char, methods to do query augmentation
        weighted: bool, whether to do weighted penalization
        uncertainty: char, method to estimate the similarity of the data pairs
        initial: char, method to initialize the cluster label, default is random
        penalized: bool, whether to penalize the metric matrx when budget is exhausted
        lambd: float, unnormalized MDSP penalty weight
        gamma: float, unnormalized selective penalty weight
        num_p: int, number of features to penalize
        n_tree: int, number of trees in the random forest
        diag: bool, whether to train diagonal metric matrix
        true_H: bool, whether to use the true H (for test purpose)
        """
        self.penalized = penalized
        self.diag = diag
        self.true_H = true_H
        self.clusterer = clusterer
        self.sequential_constraints = []
        self.true_label = None
        self.impute_method = impute_method
        self.weighted = weighted
        self.uncertainty = uncertainty
        self.initial = initial
        self.fit_count = 0
        self.lambd=lambd
        self.gamma=gamma
        self.num_p = num_p
        self.n_tree = n_tree
        self.hist_labels = dict() # record the fitted labels of each step, keys=number of constraints so far
        self.hist_labels_penalize = dict()
        self.hist_A = dict()
        self.hist_A_penalize = dict()
        self.hist_nc = list() # record number of queries after each step
        self.hist_p_idx = dict()
        self.hist_ml = dict()
        self.hist_cl = dict()
        
        if self.impute_method == 'default':
            print('default label')

        elif self.impute_method == 'true_label':
            print('true label')

        elif self.impute_method == 'random':
            print('random label')
            
    def penalized_fit(self, X, ml, cl):
        sequential_average = self.sequential_sum / self.fit_count
        tmp = [(u, i) for i, u in enumerate(sequential_average)]
        heapq.heapify(tmp)
        p_idx = [i for (u, i) in heapq.nsmallest(self.num_p, tmp)]   
        print('num_p=', self.num_p)
        print('p_idx is', p_idx)
        self.hist_p_idx[len(ml)+ len(cl)] = p_idx
        # use number of true constraints to represent the uncertainty of the inferred pair
        n, p = X.shape
        uncertainty_weight = 1
        while True:
            try:
                self.clusterer.fit(X, y=self.true_label, ml=ml, cl=cl, diag=self.diag, lambd=self.lambd*uncertainty_weight, gamma=self.gamma, penalize_idx=p_idx, true_H=self.true_H, verbose=False)
                break
            except EmptyClustersException:
                print('Empty cluster')
        
            
    def get_true_label(self, true_label):
        self.true_label = true_label
        
    def fit(self, X, oracle=None, request_nc=None):
        n, p = X.shape
        ml, cl = [], []
        neighborhoods = []
        self.sequential_sum = np.zeros(p)
        
        K = self.clusterer.n_clusters
        # initial
        if self.initial == 'default':
            x_i = np.random.choice(list(range(n)))
            neighborhoods.append([x_i])
            
        else:
            print('Initialize with clustering')
            while True:
                try:
                    self.clusterer.fit(X, ml=[], cl=[])
                    break
                except EmptyClustersException:
                    print('Empty Cluster')
                    
            for k in range(K):
                id_k = np.arange(n)[self.clusterer.labels_==k]
                tmp = sorted(id_k, key=lambda i: self.clusterer._dist(X[i], self.clusterer.cluster_centers_[k], np.eye(p))) # closest point to the cluster center
                if len(neighborhoods):
                    for nb in neighborhoods:
                        cl.append([nb[0], tmp[0]])
                neighborhoods.append([tmp[0]])
                
        while True:
            try:
                t0 = time.time()
                # dont't change order here, penalized first, then unpenalized version
                if self.penalized:
                    if oracle.queries_cnt in request_nc and oracle.queries_cnt not in self.hist_labels_penalize.keys():
                        self.penalized_fit(X, ml, cl)
                        self.hist_labels_penalize[oracle.queries_cnt] = self.clusterer.labels_
                        if not (self.lambd == 0 and self.gamma == 0):
                            self.hist_A_penalize[oracle.queries_cnt] = self.clusterer.A
                            
                while True:
                    try:
                        uncertainty_weight = 1
                        if self.lambd == 0 and self.gamma == 0:
                            self.clusterer.fit(X, ml=ml, cl=cl) # for MPCKmeans
                            if hasattr(self.clusterer, 'A'):
                                self.hist_A[oracle.queries_cnt] = self.clusterer.A
                            
                        else:
                            
                            self.clusterer.fit(X, ml=ml, cl=cl, lambd=self.lambd*uncertainty_weight, gamma=0) # unpenalized version for the proposed method
                            
                            self.hist_A[oracle.queries_cnt] = self.clusterer.A
                        break
                    except EmptyClustersException:
                        print('Empty cluster')
                        
                self.hist_labels[oracle.queries_cnt] = self.clusterer.labels_
                t1 = time.time()
                print('clusterer fitting costs: %2.3f seconds' % (t1-t0))
                self.fit_count += 1
                # compute the rank of weights
                if hasattr(self.clusterer, 'A'):
                    a = np.diag(self.clusterer.A)
                    tmp =  sorted(list(enumerate(a)), key=lambda x: x[1])
                    tmp2 = [ (tup[0], i) for i, tup in enumerate(tmp)]
                    a_rank = [r for (_, r) in sorted(tmp2)]
                    self.sequential_sum += a_rank
                      
                added_constraints = []
                x_i, p_i = self._most_informative(X, self.clusterer, neighborhoods)
                t3 = time.time()
                print("query cost: %2.3f" % (t3-t1))
                sorted_neighborhoods = list(zip(*reversed(sorted(zip(p_i, neighborhoods)))))[1]
                must_link_found = False
                t2 = time.time()
                print("Time cost for a single loop: %2.3f" % (t2-t0))
                for neighborhood in sorted_neighborhoods:

                    must_linked = oracle.query(x_i, neighborhood[0])
                    if must_linked:
                        for x_j in neighborhood:
                            ml.append([x_i, x_j])

                        for other_neighborhood in neighborhoods:
                            if neighborhood != other_neighborhood: # key part: generate more constraints with only one query
                                for x_j in other_neighborhood:
                                    cl.append([x_i, x_j])
                                    added_constraints.append([x_i, x_j])
                        neighborhood.append(x_i)
                        must_link_found = True
                        break
                    
                    else:
                        # check if we need to train the metric before querying next pair.
                        if self.penalized:
                            if oracle.queries_cnt in request_nc and oracle.queries_cnt not in self.hist_labels_penalize.keys():
                                print('extra penalized fitting at nc=%d' % oracle.queries_cnt)
                                self.penalized_fit(X, ml, cl)
                                self.hist_labels_penalize[oracle.queries_cnt] = self.clusterer.labels_  
                                if not (self.lambd == 0 and self.gamma == 0):
                                    self.hist_A_penalize[oracle.queries_cnt] = self.clusterer.A
                                    

                        if oracle.queries_cnt in request_nc and oracle.queries_cnt not in self.hist_labels.keys():
                            print('extra fitting at nc=%d' % oracle.queries_cnt)
                            while True:
                                try:
                                    if self.lambd == 0 and self.gamma == 0:
                                        self.clusterer.fit(X, ml=ml, cl=cl) # for MPCKmeans
                                        if hasattr(self.clusterer, 'A'):
                                            self.hist_A[oracle.queries_cnt] = self.clusterer.A
                                       
                                    else:    
                                        uncertainty_weight = 1
                                        self.clusterer.fit(X, ml=ml, cl=cl, lambd=self.lambd*uncertainty_weight, gamma=0) # unpenalized version
                                        self.hist_A[oracle.queries_cnt] = self.clusterer.A
                                    break
                                except EmptyClustersException:
                                    print('Empty cluster')
                            self.hist_labels[oracle.queries_cnt] = self.clusterer.labels_

                if not must_link_found:
                    for neighborhood in neighborhoods:
                        for x_j in neighborhood:
                            cl.append([x_i, x_j])
                            added_constraints.append([x_i, x_j])
                    neighborhoods.append([x_i])
                if len(added_constraints):
                    self.sequential_constraints.append(added_constraints)
                print("nc=%d" % oracle.queries_cnt)
                self.hist_nc += [oracle.queries_cnt]
                self.hist_ml[oracle.queries_cnt] = ml
                self.hist_cl[oracle.queries_cnt] = cl
            except MaximumQueriesExceeded:               
                break

        self.pairwise_constraints_ = ml, cl

        return self

    def _most_informative(self, X, clusterer, neighborhoods):
        def binary_entropy(p):
            return - (p * np.log2( max(p, 1e-4) ) + (1 - p) * np.log2( max(1 - p, 1e-4) ))
        def _entropy_change(p, i):
            """
            Parameters
            ----------
            p : n x K matrix, p[i, k] is the probability of X_i belongs to neighborhood k
            i: index of X
            Returns
            -------
            Expectated entropy change of specifying the membership of X_i

            """
            n, K = p.shape
            res = 0
            for j in range(n):
                if j != i :
                    q_before = p[i] @ p[j]
                    for k in range(K):
                        p_i = np.zeros(K)
                        p_i[k] = 1
                        q_after = p_i @ p[j]
                        res += p[i, k] * (binary_entropy(q_before) - binary_entropy(q_after))
            return res
        
        n = X.shape[0]
        l = len(neighborhoods)
        K = clusterer.n_clusters

        neighborhoods_union = set()
        neighborhood_idx = dict() # record the index of the neighborhood that the sample belongs to
        for idx, neighborhood in enumerate(neighborhoods):
            for i in neighborhood:
                neighborhoods_union.add(i)
                neighborhood_idx[i] = idx
                

        unqueried_indices = set(range(n)) - neighborhoods_union

        # TODO if there is only one neighborhood then choose the point randomly?
        if l <= 1:
            return np.random.choice(list(unqueried_indices)), [1]
                

        if self.uncertainty == 'random_forest':
            # Try using the random cluster assignment to see if the clustering result matters here
            n_estimators = self.n_tree
            rf = RandomForestClassifier(n_estimators=n_estimators)
            
            t0 = time.time()
            if self.impute_method == 'default':
                rf.fit(X, clusterer.labels_)
            elif self.impute_method == 'true':
                rf.fit(X, self.true_label)
            elif self.impute_method == 'random':
                rf.fit(X, np.random.choice(K, n))
            t1 = time.time()
            print('fitting rf costs: %2.3f seconds' % (t1-t0))
            # Compute the similarity matrix
            leaf_indices = rf.apply(X)
            S = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    S[i, j] = (leaf_indices[i,] == leaf_indices[j,]).sum()
            S = S / n_estimators
            
            p = np.empty((n, l))
            uncertainties = np.zeros(n)
            expected_costs = np.ones(n)
            
            # For each point that is not in any neighborhood...
            # TODO iterate only unqueried indices
            
            neighborhood_center = [np.mean(X[neighborhoods[n_i], :], 0) for n_i in range(l)] # calculate the mean of the neighborhoods
            distance_weight = np.empty((n, l))
            for x_i in range(n):
                if not x_i in neighborhoods_union:
                    for n_i in range(l):
                        p[x_i, n_i] = (S[x_i, neighborhoods[n_i]].sum() / len(neighborhoods[n_i]))
                        distance_weight[x_i, n_i] = np.linalg.norm(X[x_i] - neighborhood_center[n_i])
                    
                    # If the point is not similar to any neighborhood set equal probabilities of belonging to each neighborhood
                    if np.all(p[x_i,] == 0):
                        p[x_i,] = np.ones(l)
    
                    p[x_i,] = p[x_i,] / p[x_i,].sum()
                    distance_weight[x_i, ] = distance_weight[x_i, ] / (distance_weight[x_i, ].sum() + 1e-6)
    
                    if not np.any(p[x_i,] == 1):
                        positive_p_i = p[x_i, p[x_i,] > 0]
                        if not self.weighted:
                            uncertainties[x_i] = -(positive_p_i * np.log2(positive_p_i) ).sum() # weighted by the distance
                        else:
                            part_weight = distance_weight[x_i, p[x_i, ] >0]
                            uncertainties[x_i] = -(positive_p_i * np.log2(positive_p_i)*part_weight).sum() # weighted by the distance
                        expected_costs[x_i] = (positive_p_i * range(1, len(positive_p_i) + 1)).sum()
                    else:
                        uncertainties[x_i] = 0
                        expected_costs[x_i] = 1  # ?
                else:
                    tmp = np.zeros(l)
                    tmp[neighborhood_idx[x_i]] = 1
                    p[x_i, ] = tmp
                        
            # quantify the entropy change in terms of the pairs
            if self.weighted:
                expected_entropy_change = np.ones(n)
                for x_i in range(n):
                    expected_entropy_change[x_i] = _entropy_change(p, x_i)
                                        
        else:
            # use distance to estimate uncertainty instead of random forest
            p = np.empty((n, l))
            uncertainties = np.zeros(n)
            expected_costs = np.ones(n)
            neighborhood_center = [np.mean(X[neighborhoods[n_i], :], 0) for n_i in range(l)] # calculate the mean of the neighborhoods
            for x_i in range(n):
                if not x_i in neighborhoods_union:
                    tmp = [clusterer._dist(X[x_i], neighborhood_center[n_i], clusterer.A) for n_i in range(l)]
                    p[x_i, ] = tmp / np.sum(tmp)
                    if np.all(p[x_i, ] == 0):
                        p[x_i, ] = np.ones(l) / l
    
                    if not np.any(p[x_i, ] == 1):
                        positive_p_i = p[x_i, p[x_i,] > 0]                  
                        uncertainties[x_i] = -(positive_p_i * np.log2(positive_p_i) ).sum() # weighted by the distance
                        expected_costs[x_i] = (positive_p_i * range(1, len(positive_p_i) + 1)).sum()
                
                    else:
                        uncertainties[x_i] = 0
                        expected_costs[x_i] = 1 
        normalized_uncertainties = uncertainties / expected_costs
        if self.weighted:
            most_informative_i = np.argmax(expected_entropy_change)
        else:
            most_informative_i = np.argmax(normalized_uncertainties)
        return most_informative_i, p[most_informative_i]

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

def _most_informative(self, X, clusterer, neighborhoods):
        n = X.shape[0]
        l = len(neighborhoods)

        neighborhoods_union = set()
        for neighborhood in neighborhoods:
            for i in neighborhood:
                neighborhoods_union.add(i)

        unqueried_indices = set(range(n)) - neighborhoods_union

        # TODO if there is only one neighborhood then choose the point randomly?
        if l <= 1:
            return np.random.choice(list(unqueried_indices)), [1]

        # Learn a random forest classifier
        n_estimators = 50
        rf = RandomForestClassifier(n_estimators=n_estimators)
        rf.fit(X, clusterer.labels_)

        # Compute the similarity matrix
        leaf_indices = rf.apply(X)
        S = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                S[i, j] = (leaf_indices[i,] == leaf_indices[j,]).sum()
        S = S / n_estimators

        p = np.empty((n, l))
        uncertainties = np.zeros(n)
        expected_costs = np.ones(n)

        # For each point that is not in any neighborhood...
        # TODO iterate only unqueried indices
        for x_i in range(n):
            if not x_i in neighborhoods_union:
                for n_i in range(l):
                    p[x_i, n_i] = (S[x_i, neighborhoods[n_i]].sum() / len(neighborhoods[n_i]))

                # If the point is not similar to any neighborhood set equal probabilities of belonging to each neighborhood
                if np.all(p[x_i,] == 0):
                    p[x_i,] = np.ones(l)

                p[x_i,] = p[x_i,] / p[x_i,].sum()

                if not np.any(p[x_i,] == 1):
                    positive_p_i = p[x_i, p[x_i,] > 0]
                    uncertainties[x_i] = -(positive_p_i * np.log2(positive_p_i)).sum()
                    expected_costs[x_i] = (positive_p_i * range(1, len(positive_p_i) + 1)).sum()
                else:
                    uncertainties[x_i] = 0
                    expected_costs[x_i] = 1  # ?

        normalized_uncertainties = uncertainties / expected_costs

        most_informative_i = np.argmax(normalized_uncertainties)
        return most_informative_i, p[most_informative_i]


class NPU_old:
    def __init__(self, clusterer=None, impute_method='default', weighted=False, uncertainty='random_forest', initial='default', penalized=False, lambd=0, gamma=0, num_p=0, n_tree=50, diag=True, true_H=False, **kwargs):
        """
        NPU method to query (Xiong, 2014)

        clusterer: char, method to do semi-supervised clustering, default is PCKmeans
        impute_method: char, methods to do query augmentation
        weighted: bool, whether to do weighted penalization
        uncertainty: char, method to estimate the similarity of the data pairs
        initial: char, method to initialize the cluster label, default is random
        penalized: bool, whether to penalize the metric matrx when budget is exhausted
        lambd: float, unnormalized MDSP penalty weight
        gamma: float, unnormalized selective penalty weight
        num_p: int, number of features to penalize
        n_tree: int, number of trees in the random forest
        diag: bool, whether to train diagonal metric matrix
        true_H: bool, whether to use the true H (for test purpose)
        """
        self.penalized = penalized
        self.diag = diag
        self.true_H = true_H
        self.clusterer = clusterer
        self.sequential_constraints = []
        self.true_label = None
        self.impute_method = impute_method
        self.weighted = weighted
        self.uncertainty = uncertainty
        self.initial = initial
        self.fit_count = 0
        self.lambd=lambd
        self.gamma=gamma
        self.num_p = num_p
        self.n_tree = n_tree
        self.hist_labels = dict() # record the fitted labels of each step, keys=number of constraints so far
        self.hist_labels_penalize = dict()
        self.hist_A = dict()
        self.hist_A_penalize = dict()
        self.hist_nc = list()
        if self.impute_method == 'default':
            print('default label')

        elif self.impute_method == 'true_label':
            print('true label')

        elif self.impute_method == 'random':
            print('random label')
            
    def penalized_fit(self, X, ml, cl):
        sequential_average = self.sequential_sum / self.fit_count
        tmp = [(u, i) for i, u in enumerate(sequential_average)]
        heapq.heapify(tmp)
        p_idx = [i for (u, i) in heapq.nsmallest(self.num_p, tmp)]   
        print('num_p=', self.num_p)
        print('p_idx is', p_idx)
        # use number of true constraints to represent the uncertainty of the inferred pair
        n, p = X.shape
        # uncertainty_weight = (len(ml) + len(cl)) / (n*(n-1) / 2)
        uncertainty_weight = 1
        while True:
            try:
                self.clusterer.fit(X, y=self.true_label, ml=ml, cl=cl, diag=self.diag, lambd=self.lambd*uncertainty_weight, gamma=self.gamma, penalize_idx=p_idx, true_H=self.true_H, verbose=False)
                break
            except EmptyClustersException:
                print('Empty cluster')
        
            
    def get_true_label(self, true_label):
        self.true_label = true_label
        
    def fit(self, X, oracle=None, request_nc=None):
        n, p = X.shape
        ml, cl = [], []
        neighborhoods = []
        self.sequential_sum = np.zeros(p)
        
        K = self.clusterer.n_clusters
        # initial
        if self.initial == 'default':
            x_i = np.random.choice(list(range(n)))
            neighborhoods.append([x_i])
            
        else:
            print('Initialize with clustering')
            while True:
                try:
                    self.clusterer.fit(X, ml=[], cl=[])
                    break
                except EmptyClustersException:
                    print('Empty Cluster')
                    
            for k in range(K):
                id_k = np.arange(n)[self.clusterer.labels_==k]
                tmp = sorted(id_k, key=lambda i: self.clusterer._dist(X[i], self.clusterer.cluster_centers_[k], np.eye(p))) # closest point to the cluster center
                if len(neighborhoods):
                    for nb in neighborhoods:
                        cl.append([nb[0], tmp[0]])
                neighborhoods.append([tmp[0]])
                
        while True:
            try:
                t0 = time.time()
                # dont't change order here, penalized first, then unpenalized version
                if self.penalized:
                    if oracle.queries_cnt in request_nc and oracle.queries_cnt not in self.hist_labels_penalize.keys():
                        self.penalized_fit(X, ml, cl)
                        self.hist_labels_penalize[oracle.queries_cnt] = self.clusterer.labels_
                        if not (self.lambd == 0 and self.gamma == 0):
                            self.hist_A_penalize[oracle.queries_cnt] = self.clusterer.A

                while True:
                    try:
                        
                        # uncertainty_weight = (len(ml) + len(cl)) / (n*(n-1) / 2)
                        uncertainty_weight = 1
                        if self.lambd == 0 and self.gamma == 0:
                            self.clusterer.fit(X, ml=ml, cl=cl) # for MPCKmeans
                            if hasattr(self.clusterer, 'A'):
                                self.hist_A[oracle.queries_cnt] = self.clusterer.A
                            
                        else:
                            
                            self.clusterer.fit(X, ml=ml, cl=cl, lambd=self.lambd*uncertainty_weight, gamma=0) # unpenalized version for the proposed method
                            
                            self.hist_A[oracle.queries_cnt] = self.clusterer.A
                        break
                    except EmptyClustersException:
                        print('Empty cluster')
                        
                self.hist_labels[oracle.queries_cnt] = self.clusterer.labels_
                t1 = time.time()
                print('clusterer fitting costs: %2.3f seconds' % (t1-t0))
                self.fit_count += 1
                # compute the rank of weights
                if hasattr(self.clusterer, 'A'):
                    a = np.diag(self.clusterer.A)
                    tmp =  sorted(list(enumerate(a)), key=lambda x: x[1])
                    tmp2 = [ (tup[0], i) for i, tup in enumerate(tmp)]
                    a_rank = [r for (_, r) in sorted(tmp2)]
                    self.sequential_sum += a_rank
                      
                added_constraints = []
                x_i, p_i = self._most_informative(X, self.clusterer, neighborhoods)

                sorted_neighborhoods = list(zip(*reversed(sorted(zip(p_i, neighborhoods)))))[1]
                must_link_found = False
                
                for neighborhood in sorted_neighborhoods:

                    must_linked = oracle.query(x_i, neighborhood[0])
                     
                    # print(oracle.queries_cnt)
                    if must_linked:
                        # TODO is it necessary? this preprocessing is part of the clustering algorithms
                        for x_j in neighborhood:
                            ml.append([x_i, x_j])

                        for other_neighborhood in neighborhoods:
                            if neighborhood != other_neighborhood: # key part: generate more constraints with only one query
                                for x_j in other_neighborhood:
                                    cl.append([x_i, x_j])
                                    added_constraints.append([x_i, x_j])
                        neighborhood.append(x_i)
                        must_link_found = True
                        break
                    
                    else:

                        if self.penalized:
                            if oracle.queries_cnt in request_nc and oracle.queries_cnt not in self.hist_labels_penalize.keys():
                                print('extra penalied fitting at nc=%d' % oracle.queries_cnt)
                                # uncertainty_weight = (len(ml) + len(cl)) / (n*(n-1) / 2)
                                self.penalized_fit(X, ml, cl)
                                self.hist_labels_penalize[oracle.queries_cnt] = self.clusterer.labels_  
                                if not (self.lambd == 0 and self.gamma == 0):
                                    self.hist_A_penalize[oracle.queries_cnt] = self.clusterer.A
                                
                        if oracle.queries_cnt in request_nc and oracle.queries_cnt not in self.hist_labels.keys():
                        # nc = request_nc[request_nc_idx] # output of number of constraints required by the user
                        # if oracle.queries_cnt == nc:
                            print('extra fitting at nc=%d' % oracle.queries_cnt)
                            while True:
                                try:
                                    # uncertainty_weight = (len(ml) + len(cl)) / (n*(n-1) / 2)
                                    if self.lambd == 0 and self.gamma == 0:
                                        self.clusterer.fit(X, ml=ml, cl=cl) # for MPCKmeans
                                        if hasattr(self.clusterer, 'A'):
                                            self.hist_A[oracle.queries_cnt] = self.clusterer.A
                                       
                                    else:    
                                        uncertainty_weight = 1
                                        self.clusterer.fit(X, ml=ml, cl=cl, lambd=self.lambd*uncertainty_weight, gamma=0) # unpenalized version
                                        self.hist_A[oracle.queries_cnt] = self.clusterer.A
                                    break
                                except EmptyClustersException:
                                    print('Empty cluster')
                            self.hist_labels[oracle.queries_cnt] = self.clusterer.labels_                    

                        # TODO should we add the cannot-link in case the algorithm stops before it queries all neighborhoods?

                if not must_link_found:
                    for neighborhood in neighborhoods:
                        for x_j in neighborhood:
                            cl.append([x_i, x_j])
                            added_constraints.append([x_i, x_j])
                    neighborhoods.append([x_i])
                if len(added_constraints):
                    # print('number of added constraints: %d' % len(added_constraints))
                    self.sequential_constraints.append(added_constraints)
                print("nc=%d" % oracle.queries_cnt)
                self.hist_nc += [oracle.queries_cnt]
            except MaximumQueriesExceeded:               
                break

        self.pairwise_constraints_ = ml, cl

        return self

    def _most_informative(self, X, clusterer, neighborhoods):
        n = X.shape[0]
        l = len(neighborhoods)

        neighborhoods_union = set()
        for neighborhood in neighborhoods:
            for i in neighborhood:
                neighborhoods_union.add(i)

        unqueried_indices = set(range(n)) - neighborhoods_union

        # TODO if there is only one neighborhood then choose the point randomly?
        if l <= 1:
            return np.random.choice(list(unqueried_indices)), [1]

        # Learn a random forest classifier
        n_estimators = 50
        rf = RandomForestClassifier(n_estimators=n_estimators)
        rf.fit(X, clusterer.labels_)

        # Compute the similarity matrix
        leaf_indices = rf.apply(X)
        S = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                S[i, j] = (leaf_indices[i,] == leaf_indices[j,]).sum()
        S = S / n_estimators

        p = np.empty((n, l))
        uncertainties = np.zeros(n)
        expected_costs = np.ones(n)

        # For each point that is not in any neighborhood...
        # TODO iterate only unqueried indices
        for x_i in range(n):
            if not x_i in neighborhoods_union:
                for n_i in range(l):
                    p[x_i, n_i] = (S[x_i, neighborhoods[n_i]].sum() / len(neighborhoods[n_i]))

                # If the point is not similar to any neighborhood set equal probabilities of belonging to each neighborhood
                if np.all(p[x_i,] == 0):
                    p[x_i,] = np.ones(l)

                p[x_i,] = p[x_i,] / p[x_i,].sum()

                if not np.any(p[x_i,] == 1):
                    positive_p_i = p[x_i, p[x_i,] > 0]
                    uncertainties[x_i] = -(positive_p_i * np.log2(positive_p_i)).sum()
                    expected_costs[x_i] = (positive_p_i * range(1, len(positive_p_i) + 1)).sum()
                else:
                    uncertainties[x_i] = 0
                    expected_costs[x_i] = 1  # ?

        normalized_uncertainties = uncertainties / expected_costs

        most_informative_i = np.argmax(normalized_uncertainties)
        return most_informative_i, p[most_informative_i]