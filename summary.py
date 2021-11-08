# -*- coding: utf-8 -*-
"""
Helper function for result table output

@author: Yujia Deng
"""

import numpy as np
import os
def output(file, result, setting_name):
    file.write('\n'+setting_name)
    res_mean = np.mean(result, 0)
    res_sd = np.std(result, 0)
    for n in range(len(res_mean)):
        file.write('\t %2.3f(%2.3f)' % (res_mean[n], res_sd[n]))

if __name__ == '__main__':
    P2 = 30
    mu = 3
    max_nc = 300
    request_nc = range(10, max_nc+1, 10)
    reps = 30
    n_super_instances = range(10, 200, 20)

    for P1, P2, max_nc, N_per_cluster in [(5, 30, 300, 60)]:
        save_path = './simulation_high_dim_4/P1_%d_P2_%d_mu_%2.1f_N_per_cluster_%d_lambda_200/max_nc_%d/' % (P1, P2, mu, N_per_cluster, max_nc)
        result_MPCKmeans = np.zeros((reps, len(request_nc)))
        result_proposed_no_penalty = np.zeros((reps, len(request_nc)))
        result_proposed_penalty = np.zeros((reps, len(request_nc)))
        
        
        COBRA_n_query = np.zeros((reps, len(n_super_instances)))
        COBRA_ARI = np.zeros((reps, len(n_super_instances)))
        for rep in range(reps):          
            result = np.load(os.path.join(save_path, ('rep%d.npz' % (rep+1))))
            #print(result['result_MPCKmeans'])
            result_MPCKmeans[rep, :] = list(result['result_MPCKmeans'].item().values())
            result_proposed_no_penalty[rep, :] = list(result['result_proposed_no_penalty'].item().values())
            result_proposed_penalty[rep, :] = list(result['result_proposed_penalty'].item().values())
            
            
            tmp = result['result_COBRA'].item().values()
            COBRA_n_query[rep, :] = [x[0] for x in tmp]
            COBRA_ARI[rep, :] = [x[1] for x in tmp]
            
        with open(os.path.join(save_path,'result_COBRA.tsv'), 'w+') as file:
            file.write('number of super instances')
            for n in (n_super_instances):
                file.write('\t' + str(n))
            output(file, COBRA_n_query, 'average number of queries')
            output(file, COBRA_ARI, 'average ARI of COBRA')
        
        with open(os.path.join(save_path,'result.tsv'), 'w+') as file:
            file.write('number of super instances')
            for nc in request_nc:
                file.write('\t' + str(nc))
            output(file, result_MPCKmeans, 'result_ARI_NPU_MPCKmeans')
            output(file, result_proposed_no_penalty, 'result_ARI_NPU_proposed_no_penalty')
            output(file, result_proposed_penalty, 'result_ARI_NPU_proposed_penalty')


    P1 = 5
    mu = 3
    max_nc = 300
    request_nc = range(10, max_nc+1, 10)
    reps = 10
    n_super_instances = range(10, 200, 20)
    for P1, P2, max_nc, N_per_cluster  in [(5, 30, 300, 60)]:
        save_path = './simulation_high_dim_4_diagonal/P1_%d_P2_%d_mu_%2.1f_N_per_cluster_%d/max_nc_%d/' % (P1, P2, mu, N_per_cluster, max_nc)
        result_MPCKmeans = np.zeros((reps, len(request_nc)))
        result_proposed_no_penalty = np.zeros((reps, len(request_nc)))
        result_proposed_penalty = np.zeros((reps, len(request_nc)))
        
        
        COBRA_n_query = np.zeros((reps, len(n_super_instances)))
        COBRA_ARI = np.zeros((reps, len(n_super_instances)))
        for rep in range(reps):
            result = np.load(os.path.join(save_path, ('rep%d.npz' % (rep+1))))
            #print(result['result_MPCKmeans'])
            result_MPCKmeans[rep, :] = list(result['result_MPCKmeans'].item().values())
            result_proposed_no_penalty[rep, :] = list(result['result_proposed_no_penalty'].item().values())
            result_proposed_penalty[rep, :] = list(result['result_proposed_penalty'].item().values())
            
            
            tmp = result['result_COBRA'].item().values()
            COBRA_n_query[rep, :] = [x[0] for x in tmp]
            COBRA_ARI[rep, :] = [x[1] for x in tmp]
            
        with open(os.path.join(save_path,'result_COBRA.tsv'), 'w+') as file:
            file.write('number of super instances')
            for n in (n_super_instances):
                file.write('\t' + str(n))
            output(file, COBRA_n_query, 'average number of queries')
            output(file, COBRA_ARI, 'average ARI of COBRA')
        
        with open(os.path.join(save_path,'result.tsv'), 'w+') as file:
            file.write('number of super instances')
            for nc in request_nc:
                file.write('\t' + str(nc))
            output(file, result_MPCKmeans, 'result_ARI_NPU_MPCKmeans')
            output(file, result_proposed_no_penalty, 'result_ARI_NPU_proposed_no_penalty')
            output(file, result_proposed_penalty, 'result_ARI_NPU_proposed_penalty')
        
        
    # fashion pca  
    save_path = './real_data/fashion_pca_N50/max_nc_%d/' % (max_nc)
    request_nc = range(10, max_nc+1, 10)
    reps = 30
    n_super_instances = range(10, 200, 20)
    result_MPCKmeans = np.zeros((reps, len(request_nc)))
    result_proposed_no_penalty = np.zeros((reps, len(request_nc)))
    result_proposed_penalty = np.zeros((reps, len(request_nc)))


    COBRA_n_query = np.zeros((reps, len(n_super_instances)))
    COBRA_ARI = np.zeros((reps, len(n_super_instances)))
    for rep in range(reps):
        result = np.load(os.path.join(save_path, ('rep%d.npz' % (rep+1))))
        #print(result['result_MPCKmeans'])
        result_MPCKmeans[rep, :] = list(result['result_MPCKmeans'].item().values())
        result_proposed_no_penalty[rep, :] = list(result['result_proposed_no_penalty'].item().values())
        result_proposed_penalty[rep, :] = list(result['result_proposed_penalty'].item().values())
        
        
        tmp = result['result_COBRA'].item().values()
        COBRA_n_query[rep, :] = [x[0] for x in tmp]
        COBRA_ARI[rep, :] = [x[1] for x in tmp]
        
    with open(os.path.join(save_path,'result_COBRA.tsv'), 'w+') as file:
        file.write('number of super instances')
        for n in (n_super_instances):
            file.write('\t' + str(n))
        output(file, COBRA_n_query, 'average number of queries')
        output(file, COBRA_ARI, 'average ARI of COBRA')

    with open(os.path.join(save_path,'result.tsv'), 'w+') as file:
        file.write('number of super instances')
        for nc in request_nc:
            file.write('\t' + str(nc))
        output(file, result_MPCKmeans, 'result_ARI_NPU_MPCKmeans')
        output(file, result_proposed_no_penalty, 'result_ARI_NPU_proposed_no_penalty')
        output(file, result_proposed_penalty, 'result_ARI_NPU_proposed_penalt')
    