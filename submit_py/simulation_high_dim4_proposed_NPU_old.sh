WALLTIME=200:00:00
NUM_CORE=1
QUEUE_NAME=stat-qu
P1s+=" 10"
P2s+=" 30"
mu=3
N_per_clusters+=" 100"
max_nc=1000
reps=`seq 1 1 20`

for P1 in ${P1s}
do
for N_per_cluster in ${N_per_clusters}
do
for P2 in ${P2s}
do
for rep in ${reps}
do
file_name=${P1}_${P2}_${mu}_${N_per_cluster}_${max_nc}_${rep}
result_file=../python_file/simulation_high_dim_4/P1_${P1}_P2_${P2}_mu_${mu}.0_N_per_cluster_${N_per_cluster}_entropy_change_/max_nc_${max_nc}/rep${rep}.npz
if [ ! -f "$result_file" ]; then
	cat >  ${file_name}.pbs << EOF
#!/bin/bash
#PBS -l nodes=1:ppn=${NUM_CORE}
#PBS -l naccesspolicy=shared
#PBS -l walltime=${WALLTIME}
#PBS -N ${file_name}
#PBS -q ${QUEUE_NAME}
#PBS -j oe
##PBS -m bae
#PBS -M yujiad2@illinois.edu
#PBS -W group_list=stat-qu

cd /home/$USER/project-stat/active_learning/python_file
module load anaconda/3
source activate my.anaconda
python3 sequential_output_simulation_proposed_NPU_old.py ${P1} ${P2} ${mu} ${N_per_cluster} ${max_nc} ${rep}

EOF

qsub ${file_name}.pbs
echo ${file_name}
rm ${file_name}.pbs
fi
done
done
done
done