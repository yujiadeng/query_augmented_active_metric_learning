"""
Generate Table 1, 2 and 3 based on the pre-run simulation results. The generated table are saved in the ./table folder.
"""

import numpy as np
import os
import matplotlib.pyplot as plt
def output(file, result, setting_name):
    file.write('\n'+setting_name)
    res_mean = np.mean(result, 0)
    res_sd = np.std(result, 0)
    for n in range(len(res_mean)):
        file.write('\t %2.3f(%2.3f)' % (res_mean[n], res_sd[n]))
if not os.path.exists('./figure'):
    os.makedirs('./figure')

table_path = './table'
if not os.path.exists(table_path):
    os.makedirs(table_path)

for data_name in ['P1_5_P2_30_mu_3.0_N_per_cluster_60', 'P1_10_P2_30_mu_3.0_N_per_cluster_30']:
    request_nc = range(10, 310, 10)
    n_super_instances = range(10, 200, 20)
    # read in the data of proposed method and MPCKmeans
    save_path_entropy_change = './results_collection/simulation_high_dim_4/%s/max_nc_300/' % data_name
    save_path_propNPU = './results_collection/simulation_high_dim_4/%s/max_nc_300/comparison/AQM_NPU' % data_name
    save_path_comparison = './results_collection/simulation_high_dim_4/%s/max_nc_300/comparison/' % data_name
    save_path_MPCKmeans = './results_collection/simulation_high_dim_4/%s/max_nc_300/comparison/MPCKmeans' % data_name
    save_path_COBRAS = './results_collection/simulation_high_dim_4/%s/max_nc_300/comparison/COBRAS' % (data_name)
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
        result3 = np.load(os.path.join(save_path_entropy_change, ('rep%d.npz' % (rep+1))), allow_pickle=True)
        res_proposed[rep, :] = list(result3['result_proposed_penalty'].item().values())

        
        result1 = np.load(os.path.join(save_path_propNPU, ('rep%d.npz' % (rep+1))), allow_pickle=True)
        res_propNPU[rep, :] = list(result1['result_proposed_penalty'].item().values())

        result2 = np.load(os.path.join(save_path_comparison, ('rep%d.npz' % (rep+1))), allow_pickle=True)
        res_PCKmeans[rep, :] = list(result2['result_PCKmeans'].item().values())
        res_COPKmeans[rep, :] = list(result2['result_COPKmeans'].item().values())

        result4 = np.load(os.path.join(save_path_MPCKmeans, ('rep%d.npz' % (rep+1))), allow_pickle=True)
        res_MPCKmeans[rep, :] = list(result4['result_MPCKmeans'].item().values()) 

        tmp = result4['result_COBRA'].item().values()
        COBRA_n_query[rep, :] = [x[0] for x in tmp][:len(n_super_instances)]
        COBRA_ARI[rep, :] = [x[1] for x in tmp][:len(n_super_instances)]
        

    mean_MPCKmeans = np.mean(res_MPCKmeans, 0)
    mean_proposed = np.mean(res_proposed, 0)
    mean_PCKmeans = np.mean(res_PCKmeans, 0)
    mean_COPKmeans = np.mean(res_COPKmeans, 0)
    mean_propNPU = np.mean(res_propNPU, 0)
    mean_COBRAS = np.mean(res_COBRAS, 0)
    plt.figure(figsize=(8, 6))
    
    plt.plot(request_nc, mean_proposed, color='r', label='AQM_MEE', linewidth=2)
    plt.plot(request_nc, mean_propNPU, color='r', label='AQM+NPU', linewidth=1, linestyle='--')
    plt.plot(request_nc, mean_MPCKmeans, color='b', label='MPCKmeans', linewidth=1)
    plt.plot(request_nc, mean_COPKmeans, color='g', label='COPKmeans', linewidth=1)
    plt.plot(request_nc, mean_PCKmeans, color='cyan', label='PCKmeans', linewidth=1)
    plt.plot(request_nc, mean_COBRAS, color='darkorange', label='COBRAS', linewidth=1)

    mean_COBRA_query = np.mean(COBRA_n_query, 0)
    mean_COBRA_ARI  = np.mean(COBRA_ARI, 0)
    plt.plot(mean_COBRA_query, mean_COBRA_ARI, 'x', color='black', label='COBRA')
    plt.legend(loc=2)
    plt.xlim((0, 300))
    if data_name == 'P1_10_P2_30_mu_3.0_N_per_cluster_30':
        plt.ylim((0, 0.4))
    plt.xlabel('Number of queries', fontsize=16)
    plt.ylabel('ARI', fontsize=16)
    plt.savefig('./figure/results_collection_%s_comparison.png' % data_name, bbox_size='tight')
    
    if data_name == 'P1_5_P2_30_mu_3.0_N_per_cluster_60':
        table_idx = '1'
    else:
        table_idx = '2'

    with open(os.path.join(table_path,'Table_%s_1.tsv' % table_idx ), 'w+') as file:
        file.write('number of super instances')
        output_request_nc = range(60, 310, 60)
        for nc in output_request_nc:
            file.write('\t' + str(nc))
        column_idx = [ int(x/10) - 1 for x in output_request_nc]
        output(file, res_PCKmeans[:, column_idx], 'PCKmeans + NPU')
        output(file, res_COPKmeans[:, column_idx], 'COPKmeans + NPU')
        output(file, res_MPCKmeans[:, column_idx], 'MPCKmeans + NPU')
        output(file, res_COBRAS[:, column_idx], 'COBRAS + NPU')
        output(file, res_propNPU[:, column_idx], 'AQM + NPU')
        output(file, res_proposed[:, column_idx], 'AQM + MEE')

    with open(os.path.join(table_path,'Table_%s_2.tsv' % table_idx ), 'w+') as file:
        file.write('number of super instances')
        if data_name == 'P1_5_P2_30_mu_3.0_N_per_cluster_60':
            output_n_super_instances = range(10, 180, 40)
            column_idx = [ int((x - 10) / 40 * 2) for x in output_n_super_instances]
        else:
            output_n_super_instances = range(10, 100, 20)
            column_idx = [ int((x - 10) / 20) for x in output_n_super_instances]
        for n in (output_n_super_instances):
            file.write('\t' + str(n))
        output(file, COBRA_n_query[:,column_idx], 'Number of queries')
        output(file, COBRA_ARI[:,column_idx], 'ARI of COBRA')

# Table 3    
reps = 10
request_nc = range(10, 310, 10)
n_super_instances = range(10, 200, 20)
save_path_proposed = './results_collection/simulation_sphere/K_5_P1_100_P2_400_r_5.0_N_per_cluster_50_entropy_change_lambda_200/max_nc_300'
save_path_comparison = './results_collection/simulation_sphere/K_5_P1_100_P2_400_r_5.0_N_per_cluster_50_entropy_change_lambda_200/max_nc_300/comparison'
save_path_COBRAS = save_path_comparison

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
    result3 = np.load(os.path.join(save_path_proposed, ('rep%d.npz' % (rep+1))), allow_pickle=True)
    res_proposed[rep, :] = list(result3['result_proposed_penalty'].item().values())

    # result1 = np.load(os.path.join(save_path_comparison, ('rep%d.npz' % (rep+1))), allow_pickle=True)
    # res_propNPU[rep, :] = list(result1['result_proposed_penalty'].item().values())

    result2 = np.load(os.path.join(save_path_comparison, ('rep%d.npz' % (rep+1))), allow_pickle=True)
    res_PCKmeans[rep, :] = list(result2['result_PCKmeans'].item().values())
    res_COPKmeans[rep, :] = list(result2['result_COPKmeans'].item().values())

    result4 = np.load(os.path.join(save_path_proposed, ('rep%d.npz' % (rep+1))), allow_pickle=True)
    res_MPCKmeans[rep, :] = list(result4['result_MPCKmeans'].item().values()) 

    tmp = result4['result_COBRA'].item().values()
    COBRA_n_query[rep, :] = [x[0] for x in tmp][:len(n_super_instances)]
    COBRA_ARI[rep, :] = [x[1] for x in tmp][:len(n_super_instances)]
    

mean_MPCKmeans = np.mean(res_MPCKmeans, 0)
mean_proposed = np.mean(res_proposed, 0)
mean_PCKmeans = np.mean(res_PCKmeans, 0)
mean_COPKmeans = np.mean(res_COPKmeans, 0)
# mean_propNPU = np.mean(res_propNPU, 0)
mean_COBRAS = np.mean(res_COBRAS, 0)

plt.figure(figsize=(8, 6))
plt.plot(request_nc, mean_proposed, color='r', label='AQM_MEE', linewidth=2)
# plt.plot(request_nc, mean_propNPU, color='r', label='AQM+NPU', linewidth=1, linestyle='--')
plt.plot(request_nc, mean_MPCKmeans, color='b', label='MPCKmeans', linewidth=1)
plt.plot(request_nc, mean_COPKmeans, color='g', label='COPKmeans', linewidth=1)
plt.plot(request_nc, mean_PCKmeans, color='cyan', label='PCKmeans', linewidth=1)
plt.plot(request_nc, mean_COBRAS, color='darkorange', label='COBRAS', linewidth=1)

mean_COBRA_query = np.mean(COBRA_n_query, 0)
mean_COBRA_ARI  = np.mean(COBRA_ARI, 0)
plt.plot(mean_COBRA_query, mean_COBRA_ARI, 'x', color='black', label='COBRA')
plt.legend(loc=2)
plt.xlim((0, 300))
plt.xlabel('Number of queries', fontsize=16)
plt.ylabel('ARI', fontsize=16)
plt.savefig('./figure/results_collection_sphere_comparison.png', bbox_size='tight')


with open(os.path.join(table_path,'Table_3_1.tsv'), 'w+') as file:
    file.write('number of super instances')
    output_request_nc = range(60, 310, 60)
    for nc in output_request_nc:
        file.write('\t' + str(nc))
    column_idx = [ int(x/10) - 1 for x in output_request_nc]
    output(file, res_PCKmeans[:, column_idx], 'PCKmeans + NPU')
    output(file, res_COPKmeans[:, column_idx], 'COPKmeans + NPU')
    output(file, res_MPCKmeans[:, column_idx], 'MPCKmeans + NPU')
    output(file, res_COBRAS[:, column_idx], 'COBRAS + NPU')
    output(file, res_proposed[:, column_idx], 'AQM + MEE')

with open(os.path.join(table_path,'Table_3_2.tsv'), 'w+') as file:
    file.write('number of super instances')
    output_n_super_instances = [30, 70, 90, 110, 150]
    column_idx = [ int((x - 10) / 20) for x in output_n_super_instances]
    for n in (output_n_super_instances):
        file.write('\t' + str(n))
    output(file, COBRA_n_query[:,column_idx], 'Number of queries')
    output(file, COBRA_ARI[:,column_idx], 'ARI of COBRA')