# Code for paper Query Augmented Active Metric Learning
This repo is for reproducing the numerical results in the paper Query Augmented Active Metric Learning.

### Environment Requirement

+ Python 3.6.8
+ Numpy 1.16.5
+ [active-semi-supervised-clustering 0.0.1](https://github.com/datamole-ai/active-semi-supervised-clustering)
+ [metric_learn 0.5.0](https://github.com/scikit-learn-contrib/metric-learn)
+ [cobra](https://bitbucket.org/toon_vc/cobra/src/master/)
+ [cobras](https://github.com/ML-KULeuven/cobras)

### File Descriptions

`datasets`: MEU-Mobile and urban land cover datasets.

`proposed_cluster.py`: proposed clusterer with Augmented Query Metric learning method (AQM).

`mynpu_metric.py`: proposed active query system using the Minimum Expected Entropy (MEE) criterion.

`sample_code.py`: a small demo showing how to apply the proposed method to do active semi-supervised clustering.

`sequential_output_*.py`, `cobras_comparison.py`, `comparison_inferred_H.py `: numerical experiment logistics.

`real_data_figures.py`, `simulation_table_figure.py`, `weights_plot_comparison.py` : result presentation and visualization.

The rest are the helper files.



### Instructions

After cloning this repo and installing the necessary libraries, please follow the instructions below to reproduce the results.

**Simulations in Section 6.1**:  

Run `comparison_inferred_H.py`. The result figures are saved as `./figure/Figure_1_*.png`.



**Simulations in Section 6.2**:  

1. **Non-sphere setting:** (Table 1 and 2)

Run `sequential_output_simulation_comparison.py`, `sequential_output_simulation_MPCKmeans.py`, `sequential_output_simulation_proposed_NPU_old.py`, and `sequential_output_simulation_entropy_change.py` for a single replication of the simulation. The simulation settings can be changed by either

+ modifying the line: `sys.argv = ...` 

or

+ commenting the line  `sys.argv = ...` and running the following in the terminal

  ```cmd
  python sequential_output_simulation_comparison.py 5 30 3 60 300 0
  ```

  The parameters in order are $p_1$, $p_2$, $\mu$, number of points per cluster, maximum queries, replication index

Alternatively, we can run multiple replications of experiments parallelly on the cluster to save time. The template shell scripts for batch job can be found in `./submit_py/` as `simulation_high_dim4_comparison.sh`, `simulation_high_dim4_proposed_NPU_old.sh`, `simulation_high_dim4_entropy_change.sh`.

After the program finishes, the results should be saved in `./results_collection/simulation_high_dim_4/`.

 

2. **Sphere setting:** (Table 3)

Run `sequential_output_simulation_sphere_comparison.py` and `sequential_output_simulation_sphere_entropy_change.py` for a single replication.

Batch job script template can be found in  `./submit_py/` as `slurm_simulation_sphere.sh`, `slurm_simulation_sphere_comparison.sh`.

After the program finishes, the results should be saved in `./results_collection/simulation_sphere/`.



**Real Data**

Run `sequential_output_real_data_comparison.py` and `sequential_output_real_data_entropy_change.py` for a single replication.

Batch job script template can be found in  `./submit_py/` as `real_data_comparison.sh`, `real_data_entropy_change.sh`.

After the program finishes, the results should be saved in `./real_data/`.

 

***COBRAS**

Run `cobras_comparison.py` to add COBRAS as a competing method to the simulation and real data settings.



### Table and Figure reproduction

To reproduce the figures and tables, the aforementioned scripts should be run for multiple replications first. **Alternatively, the pre-run numerical results can be downloaded  as `numerical_results.zip`, ** After downloading and unzipping, please put `results_collection` and `real_data` folder under the working directory. 

Below lists the files to generate the corresponding tables and figures in the paper. All figures and tables are saved in `./figure/` and `./table/`, respectively.

**Figure 1**: `comparison_inferred_H.py`. Figures are saved as `Figure_1_1.png` and `Figure_1_2.png`.

**Figure 2**: `comparison_inferred_H.py`. The Figure is saved as `Figure_2.png`.

**Figure 3**: `real_data_figures.py`. Figures are saved as `Figure_3_a.png`, `Figure_3_b.png` and `Figure_3_c.png`.

**Figure 4**: `weights_plot_comparison.py`. The Figure is saved as `Figure_4.png`.



**Table 1**: `simulation_table_figure.py`. Tables are saved as `Table_1_1.tsv` and `Table_1_2.tsv`.

**Table 2**: `simulation_table_figure.py`. Tables are saved as `Table_2_1.tsv` and `Table_2_2.tsv`.

**Table 3**: `simulation_table_figure.py`. Tables are saved as `Table_3_1.tsv` and `Table_3_2.tsv`.



### Troubleshooting

+ The `metric-learn` package may have the following error: `cannot import name 'logsumexp' from 'sklearn.utils.fixes'`. Please manually change to `scipy.special.logsumexp` instead. See https://github.com/scikit-learn-contrib/metric-learn/issues/289 for details.









