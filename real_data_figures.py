"""
Generate Figure 3 based on the pre-run results.

@author: Yujia Deng
"""

import numpy as np
import os
import matplotlib.pyplot as plt
if not os.path.exists('./figure'):
    os.makedirs('./figure')
for data_name in ['breast_cancer', 'urban_land_cover', 'MEU-Mobile']:
    request_nc = range(10, 310, 10)
    n_super_instances = range(10, 200, 20)
    # read in the data of proposed method and MPCKmeans
    save_path = './real_data/%s/max_nc_300/' % data_name
    save_path_entropy_change = './real_data/%s_entropy_change/max_nc_300/' % data_name
    save_path_COBRAS = './real_data/%s/max_nc_300/comparison/' % (data_name)
    reps = 30
    res_MPCKmeans = np.zeros((reps, len(request_nc)))
    res_proposed = np.zeros((reps, len(request_nc)))
    res_PCKmeans = np.zeros((reps, len(request_nc)))
    res_COPKmeans = np.zeros((reps, len(request_nc)))
    # proposed_method with NPU strategy
    res_propNPU = np.zeros((reps, len(request_nc)))
    
    res_COBRAS = np.load(os.path.join(save_path_COBRAS, 'COBRAS.npz'), allow_pickle=True)['results_COBRAS']
    
    
    COBRA_n_query = np.zeros((reps, len(n_super_instances)))
    COBRA_ARI = np.zeros((reps, len(n_super_instances)))
    
    for rep in range(reps):
        result1 = np.load(os.path.join(save_path, ('rep%d.npz' % (rep+1))), allow_pickle=True)
        if data_name != 'breast_cancer':
            res_MPCKmeans[rep, :] = list(result1['result_MPCKmeans'].item().values()) 
        else:
            result_MPCKmeans = np.load(os.path.join(save_path, ('rep%d_MPCKmeans.npz' % (rep+1))), allow_pickle=True)
            res_MPCKmeans[rep, :] = list(result_MPCKmeans['result_MPCKmeans'].item().values()) 
        res_propNPU[rep, :] = list(result1['result_proposed_penalty'].item().values())
        tmp = result1['result_COBRA'].item().values()
        COBRA_n_query[rep, :] = [x[0] for x in tmp][:len(n_super_instances)]
        COBRA_ARI[rep, :] = [x[1] for x in tmp][:len(n_super_instances)]

        result2 = np.load(os.path.join(save_path, ('comparison/rep%d.npz' % (rep+1))), allow_pickle=True)
        res_PCKmeans[rep, :] = list(result2['result_PCKmeans'].item().values())
        res_COPKmeans[rep, :] = list(result2['result_COPKmeans'].item().values())
        
        result3 = np.load(os.path.join(save_path_entropy_change, ('rep%d.npz' % (rep+1))), allow_pickle=True)
        res_proposed[rep, :] = list(result3['result_proposed_penalty'].item().values())
        

    mean_MPCKmeans = np.mean(res_MPCKmeans, 0)
    mean_proposed = np.mean(res_proposed, 0)
    mean_PCKmeans = np.mean(res_PCKmeans, 0)
    mean_COPKmeans = np.mean(res_COPKmeans, 0)
    mean_propNPU = np.mean(res_propNPU, 0)
    mean_COBRAS = np.mean(res_COBRAS, 0)
    plt.figure()
    plt.plot(request_nc, mean_proposed, color='r', label='Proposed', linewidth=2)

    plt.plot(request_nc, mean_MPCKmeans, color='b', label='MPCKmeans', linewidth=1)
    plt.plot(request_nc, mean_COPKmeans, color='g', label='COPKmeans', linewidth=1)
    plt.plot(request_nc, mean_PCKmeans, color='cyan', label='PCKmeans', linewidth=1)
    plt.plot(request_nc, mean_COBRAS, color='darkorange', label='COBRAS', linewidth=1)
    
    
    mean_COBRA_query = np.mean(COBRA_n_query, 0)
    mean_COBRA_ARI  = np.mean(COBRA_ARI, 0)
    plt.plot(mean_COBRA_query, mean_COBRA_ARI, 'x', color='black', label='COBRA')
    plt.legend()
    plt.xlim((0, 300))
    plt.xlabel('Number of queries')
    plt.ylabel('ARI')
    if data_name == 'breast_cancer':
        figure_name = '3_a'
    elif data_name == 'MEU-Mobile':
        figure_name = '3_b'
    elif data_name == 'urban_land_cover':
        figure_name = '3_c'
    plt.savefig('./figure/Figure_%s.png' % figure_name, bbox_size='tight')
    