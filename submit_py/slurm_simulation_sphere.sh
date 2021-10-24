WALLTIME=20:00:00
NUM_CORE=1
QUEUE_NAME=stat-qu
Ks+=" 5"
P1s+=" 100"
P2s+=" 400"
mu=5
N_per_clusters+=" 50"
max_nc=500
reps=`seq 1 1 10`

for K in ${Ks}
do
for P1 in ${P1s}
do
for N_per_cluster in ${N_per_clusters}
do
for P2 in ${P2s}
do
for rep in ${reps}
do
file_name=sphere_${P1}_${P2}_${mu}_${N_per_cluster}_${max_nc}_${rep}
result_file=../python_file/simulation_sphere/K_${K}_P1_${P1}_P2_${P2}_mu_${mu}.0_N_per_cluster_${N_per_cluster}_entropy_change_/max_nc_${max_nc}/rep${rep}.npz
if [ ! -f "$result_file" ]; then
	cat >  ${file_name}.slurm << EOF
#!/bin/bash
#
#SBATCH --time=${WALLTIME}                 # Job run time (hh:mm:ss)
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --ntasks-per-node=${NUM_CORE}             # Number of task (cores/ppn) per node
#SBATCH --job-name=${file_name}           # Name of batch job
#SBATCH --partition=stat          # Partition (queue)
#SBATCH --mail-user=yujiad2@illinois  # Send email notifications
##SBATCH --mail-type=BEGIN,END           # Type of email notifications to send
#       
#
	

cd /home/$USER/project-stat/active_learning/python_file
module load anaconda/3
source activate my.anaconda
python3 sequential_output_simulation_sphere_entropy_change.py ${K} ${P1} ${P2} ${mu} ${N_per_cluster} ${max_nc} ${rep}

EOF

sbatch ${file_name}.slurm
echo ${file_name}
rm ${file_name}.slurm
fi
done
done
done
done
done
