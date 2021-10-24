WALLTIME=100:00:00
NUM_CORE=1
QUEUE_NAME=stat-qu

data_names+="breast_cancer MEU-Mobile urban_land_cover"
max_nc=130
reps=`seq 1 1 30`

for data_name in ${data_names}
do
for rep in ${reps}
do
file_name=${data_name}_${max_nc}_${rep}
result_file=../python_file/real_data/${data_name}_entropy_change/max_nc_${max_nc}/comparison/rep${rep}.npz
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
python3 sequential_output_real_data_entropy_change.py ${data_name} ${max_nc} ${rep}

EOF

qsub ${file_name}.pbs
echo ${file_name}
rm ${file_name}.pbs
fi
done
done
