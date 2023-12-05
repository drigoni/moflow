#!/bin/bash
### Owner: Davide Rigoni

### ============ SLURM COMMANDS
#SBATCH --job-name='MoFlow-cancer-preprocessing'
#SBATCH --mail-user=davide.rigoni.1@unipd.it 		## user's email
#SBATCH --output=./cluster/out/MoFlow-cancer_v2_preprocessing.out
#SBATCH --partition=allgroups,testing
#SBATCH --nodes=1                                   # node count
#SBATCH --ntasks-per-node=1                         # total number of tasks per node
#SBATCH --cpus-per-task=8
#SBATCH --exclusive							
#SBATCH --mem=28G							
#SBATCH --time=1-00:00:00						



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
progPath="${HOME}/repository/ProgettoCancro-moflow/data/"
cd $progPath

# input config file for the script
srun python data_preprocess.py --data_name cancer

# last default print
echo 'Job done.'
echo -n 'Date: '
date
