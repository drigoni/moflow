#!/bin/bash
### Owner: Davide Rigoni

### ============ SLURM COMMANDS
#SBATCH --job-name='MoFlow-optimization-cancer_test_unnormalized'
#SBATCH --mail-user=davide.rigoni.1@unipd.it 		## user's email
#SBATCH --output=./cluster/out/MoFlow-optimization-cancer_v2_test_unnormalized.out
#SBATCH --partition=testing
#SBATCH --nodes=1                                   # node count
#SBATCH --ntasks-per-node=1                         # total number of tasks per node
#SBATCH --cpus-per-task=2		                    # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --exclusive							
#SBATCH --mem=50G							
#SBATCH --time=5-12						
#SBATCH --gres=gpu:1				


### ============ SOME PRINT COMMANDS
echo -n 'Date: '
date
echo -n 'Directory: '
pwd
echo -n 'Questo job viene eseguito sui seguenti nodi: '
echo ${SLURM_NODELIST}
echo ''
echo ''


### ============ COMMAND TO EXECUTE
# default paths'
SHAREDIR="/conf/shared-software/Singularity/CUDA/"
NODE_NAME="dellsrv1"
envPath="${HOME}/Programs/miniconda/envs/moflow/bin"
progPath="${HOME}/repository/ProgettoCancro-moflow/mflow/"
cd $progPath

# input config file for the script
CMD=$1

srun python optimize_cancer_property.py -snapshot model_snapshot_epoch_200  \
                                        --hyperparams_path moflow-params.json \
                                        --batch_size 128 \
                                        --model_dir results/cancer_v2_128_22   \
                                        --gpu 0   \
                                        --data_name cancer   \
                                        --property_name AVERAGE_GI50 \
                                        --topk 100  \
                                        --property_model_path average_gi50_big_model_unnormalized.pt \
                                        --debug false  \
                                        --topscore \
                                        --model_suffix _unnormalized




# last default print
echo 'Job done.'
echo -n 'Date: '
date
