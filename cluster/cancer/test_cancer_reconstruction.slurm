#!/bin/bash
### Owner: Davide Rigoni

### ============ SLURM COMMANDS
#SBATCH --job-name='MoFlow-reconstruction-cancer'
#SBATCH --mail-user=davide.rigoni.1@unipd.it 		## user's email
#SBATCH --output=./cluster/out/MoFlow-reconstruction-cancer_v2.out
#SBATCH --partition=allgroups,testing
#SBATCH --nodes=1                                   # node count
#SBATCH --ntasks-per-node=1                         # total number of tasks per node
#SBATCH --cpus-per-task=16		                    # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --exclusive							
#SBATCH --mem=50G							
#SBATCH --time=2-12						
####SBATCH --gres=gpu:1						


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
envPath="${HOME}/Programs/miniconda/envs/moflow/bin"
progPath="${HOME}/repository/ProgettoCancro-moflow/mflow/"
cd $progPath

# input config file for the script
CMD=$1

python generate.py --model_dir results/cancer_v2_128_22  -snapshot model_snapshot_epoch_200 --gpu  0  --data_name cancer --hyperparams-path moflow-params.json --batch-size 128  --reconstruct

# last default print
echo 'Job done.'
echo -n 'Date: '
date
