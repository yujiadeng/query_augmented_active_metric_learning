
from helper import *
import cvxpy as cp
from Step1_Impute import infer_membership_from_label
from active_semi_clustering.exceptions import EmptyClustersException
from scipy.linalg import null_space
import scipy

class proposed_clusterer:
    
    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters
        self.A = None
        
    def fit(self, X, y=None, ml=[], cl=[], diag=True, include_H=True, true_H=False, lambd=1, gamma=0, rank=1, penalize_idx=None, verbose=None):
        S = {(x1, x2) for (x1, x2) in ml}
        D = {(x1, x2) for (x1, x2) in cl}
        
        N, p = X.shape
        K = self.n_clusters
        if true_H:
            print('using true_H')
            H = np.zeros((N, K))
            for i in range(N):
                H[i, int(y[i])] = 1
        else:
            H = infer_membership_from_label(S, D, N, K)
        denom = [x if x>0 else 1 for x in np.std(X,0)]
        X = X / denom
        M_D = np.zeros((p,p))
        
        for i, j in D:
            M_D += np.outer(X[i,:]-X[j,:], X[i,:]-X[j,:])
    
        U = set(combinations(range(N),2))
        
        K = H.shape[1]
        tmp = np.sum(H*H, 0)
        center_mu = np.zeros((K, p))
        for k in range(K):
            for j in range(p):
                center_mu[k,j] = np.sum([H[i,k]**2*X[i,j] for i in range(N)] ) / tmp[k]
        
        center_vec = np.zeros(p)
        for j in range(p):
            center_vec[j] = sum([ H[i,k]**2 * (X[i,j] - center_mu[k,j])**2 for i in range(N) for k in range(K)])
              
        if diag:
            # penality on the distance to the center
    
         
            a = cp.Variable(p)
            a.value = np.repeat(1,p) #initial
            
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
            count_S = 0
            count_D = 0
            for i, j in U:
                coef = (H[i].dot(H[j]))-1/K
                if coef > 0:
                    count_S += 1
                else:
                    count_D += 1
                coef_S = max(coef * K/(K-1), 0)
                coef_D = max(-coef * K, 0)
                M_W_S += (X[i] - X[j])**2 * coef_S
                M_W_D += np.sqrt((X[i] - X[j])* (X[i] - X[j]) * coef_D)
                
            sum_S = M_S @ a
            sum_D = M_D @ cp.sqrt(a)
            sum_W_S = M_W_S @ a
            sum_W_D = M_W_D @ cp.sqrt(a)
            if include_H:
                if not penalize_idx:
                    objective = cp.Minimize(lambd * sum_S / max(len(S),1) + lambd * sum_W_S /max(count_S, 1) +  center_sum)
                else:
                    penalize_idx = list(penalize_idx)
                    objective = cp.Minimize(lambd * sum_S / max(len(S),1) + lambd * sum_W_S /max(count_S, 1) +  center_sum + gamma*cp.pnorm(a[penalize_idx], p=1))
            else:
                objective = cp.Minimize(sum_S - cp.log(sum_D))

            constraints = [0 <= a, sum_D/len(D) + sum_W_D/max(count_D, 1) >= 1]
            prob = cp.Problem(objective, constraints)
            prob.solve(warm_start=True, verbose=False)
        
            ans = np.diag(a.value/np.linalg.norm(a.value))

            if penalize_idx:
                print('penalty term:', gamma*np.linalg.norm(a.value[penalize_idx], 1))
            A = ans
        else:
            print('non-diagonal')
            def proj2ball(v):
                if np.linalg.norm(v) <= 1:
                    return v
                else:
                    return v / np.linalg.norm(v)
                
            def fS1(C2_S, v):
                return 2 * C2_S @ v
            def fD1(X, c, d, e, f, w_D, v):
                A = np.outer(v, v)
                diff1 = X[c] - X[d]
                M1 = np.einsum('ij,ik->ijk', diff1, diff1) / len(c)
                dist1 = np.sqrt(np.einsum('ijk,jk', M1, A) + 1e-6 )
                sum_deri1 = np.einsum('ijk,i->jk', M1, 1 / (dist1 + 1e-6))
                
                diff2 = X[e] - X[f]
                M2 = np.einsum('ij,ik->ijk', diff2, diff2) / len(e)
                dist2 = np.sqrt(np.einsum('ijk,jk', M2, A) + 1e-6 )
                sum_deri2 = np.einsum('ijk,i->jk', M2, 1 * np.array(w_D) / (dist2 + 1e-6))
                
                return (sum_deri1 + sum_deri2) @ v / (dist1.sum() + dist2.sum() + 1e-6)
                
            def grad_projection(grad1, grad2):
                """
                project grad1 to the complement space of grad2
                """
                grad2 = grad2 / (np.linalg.norm(grad2) + 1e-6)
                gtemp = grad1 - np.sum(grad1 * grad2) * grad2
                gtemp /= np.linalg.norm(gtemp + 1e-6)
                return gtemp
            def fD(X, c, d, e, f, w_D, v):
                A = np.outer(v, v)
                diff1 = X[c] - X[d]
                diff2 = X[e] - X[f]

                return np.log(np.sum(np.sqrt(np.sum(np.dot(diff1, A) * diff1, axis=1) + 1e-6)) / len(c)+\
                              np.sum(w_D * np.sqrt(np.sum(np.dot(diff2, A) * diff2, axis=1) + 1e-6 )) /  len(e) + 1e-6)
                    
            def rank1_update(X, S, D, H, include_H, lambd=0, gamma=0, penalize_idx=None, verbose=False, prev=None):    
                # penality on the distance to the center
                max_iter = 100
                max_proj = 10000
                convergence_threshold = 1e-3 
                a = [pair[0] for pair in S]
                b = [pair[1] for pair in S]
                c = [pair[0] for pair in D]
                d = [pair[1] for pair in D]
                num_pos = len(a)
                num_neg = len(c)
                num_samples, num_dim = X.shape
        
                error1 = error2 = 1e10
                eps = 0.01        # error-bound of iterative projection on C1 and C2
                # initialization
                if prev is None:
                    v = np.zeros(X.shape[1])
                    v[0] = 1
                else:
                    n_s = null_space(prev.T)
                    v = n_s[:, 0]
                    
                M_S = np.zeros((p, p))
                for i, j in S:
                    M_S += np.outer(X[i] - X[j], X[i] - X[j])
                
                M_W_S = np.zeros((p, p))
        #        sum_W_D = 0
                e, f, w_D = [], [], []
                count_S = 0
                count_D = 0
                for i, j in U:
                    coef = (H[i].dot(H[j]))-1/K
                    if coef > 0:
                        count_S += 1
                    else:
                        count_D += 1
                        e.append(i)
                        f.append(j)
                        coef_D = max(-coef * K, 0)
                        w_D.append(coef_D)
                    coef_S = max(coef * K/(K-1), 0)
        #            sum_W_S += (X[i] - X[j])**2 @a * coef_S
                    M_W_S += np.outer(X[i] - X[j], X[i] - X[j]) * coef_S
                    
                center_mat = np.zeros((p, p))
                for i in range(N):
                    for k in range(K):
                        center_mat += H[i, k]**2 * np.outer(X[i] - center_mu[k], X[i] - center_mu[k])
                # the matrix before the similar constraint    
                C2_S = lambd/max(len(S),1) * M_S + lambd/max(count_S, 1) * M_W_S + center_mat
                w = C2_S.ravel()
                A = np.outer(v, v)
                t = w.dot(A.ravel()) / 100.0
        
                w_norm = np.linalg.norm(w)
                w1 = w / w_norm  # make `w` a unit vector
                t1 = t / w_norm  # distance from origin to `w^T*x=t` plane
            
                cycle = 1
                alpha = 0.1  # initial step size along gradient
                        
                ## swithing grad1 and grad2?
                grad1 = fS1(C2_S, v) # gradient of similarity constraint function
                grad2 = fD1(X, c, d, e, f, w_D, v) # gradient of dissimilarity constraint function
                M = grad_projection(grad1, grad2) # gradient of fD1 orthogonal to fS1
                v_old = v.copy()
                
                for cycle in range(max_iter):
                    satisfy = False
                    for it in range(max_proj):
                        # First constraint: Similar pairs < t
                        norm_mat = mat_sqrt(C2_S/(t+1e-6))
                        ##
                        # v = np.linalg.inv(norm_mat).dot(proj2ball(norm_mat.dot(v)))
                        v = proj2ball(np.linalg.inv(norm_mat + 1e-4).dot(v))
                        if prev is not None:
                            v = (np.eye(p) - proj_mat(prev)) @ v
                        #
                        A = np.outer(v, v)
                        fDC2 = w.dot(A.ravel())
                        error2 = (fDC2 - t) / (t + 1e-6)
                        if error2 < eps:
                            satisfy = True
                            break
                    # gradient ascent
                    
                    obj_previous = fD(X, c, d, e, f, w_D, v_old)
                    obj = fD(X, c, d, e, f, w_D, v)
                    if satisfy and (obj > obj_previous or cycle == 0):
                        # If projection of 1 and 2 is successful, and such projection
                        # improves objective function, slightly increase learning rate
                        # and update from the current A.
                        alpha *= 1.05
                        v_old[:] = v
                        grad2 = fS1(C2_S, v)
                        grad1 = fD1(X, c, d, e, f, w_D, v)
                        M = grad1
                        v += alpha * M
                    else:
                        # If projection of 1 and 2 failed, or obj <= obj_previous due
                        # to projection of 1 and 2, shrink learning rate and re-update
                        # from the previous A.
                        alpha /= 2
                        grad2 = fS1(C2_S, v)
                        grad1 = fD1(X, c, d, e, f, w_D, v)
                        # M = grad_projection(grad1, grad2)
                        M = grad1
                        
                        # if prev is not None:
                            # M = (np.eye(p) - proj_mat(prev)) @ M # project to the orthogonal space of the previous directions
    #                    print('condition fail')
                        v[:] = v_old + alpha * M
                    delta = np.linalg.norm(alpha * M) / (np.linalg.norm(v_old) + 1e-6)
                    if delta < convergence_threshold:
                        break
                    if verbose:
                        print('mmc iter: %d, conv = %f, projections = %d' % (cycle, delta, it+1))
    #                    print(v)
    #                    print(np.linalg.norm(v))
                if delta > convergence_threshold:
                  converged = False
                  if verbose:
                    print('mmc did not converge, conv = %f' % (delta,))
                else:
                  converged = True
                  if verbose:
                    print('mmc converged at iter %d, conv = %f' % (cycle, delta))
                return v_old
            
            v1 = rank1_update(X, S, D, H, include_H, lambd, gamma, None, verbose)
            prev = v1.reshape(-1, 1)
            for r in range(rank-1):
                v = rank1_update(X@(np.eye(p)-proj_mat(prev)), S, D, H, include_H, lambd, gamma, None, verbose, prev)
                prev = np.column_stack((prev, v))
                
            rotate_mat = proj_mat(prev)
            weight = np.repeat(1, p)
            A = rotate_mat.dot(np.diag(weight)).dot(rotate_mat)
            
        model = PCKMeans(K)
        while True:
            try:
                model.fit(transform(X, A), ml=list(S), cl=list(D))        
                break
            except EmptyClustersException:
                print('Empty cluster')
                continue
        self.labels_, self.A = model.labels_, A 
        self.cluster_centers_ = self._get_cluster_centers(X, model.labels_)
        return self
    
    def _dist(self, x, y, A):
        "(x - y)^T A (x - y)"
        return scipy.spatial.distance.mahalanobis(x, y, A) ** 2
    
    def _get_cluster_centers(self, X, labels):
        return np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])
    
    
    