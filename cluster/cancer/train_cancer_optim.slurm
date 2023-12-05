#!/bin/bash
### Owner: Davide Rigoni

### ============ SLURM COMMANDS
#SBATCH --job-name='MoFlow-trainoptim-cancer'
#SBATCH --mail-user=davide.rigoni.1@unipd.it 		## user's email
#SBATCH --output=./cluster/out/MoFlow-trainoptimm-cancer_v2-qed-plogp.out
#SBATCH --partition=allgroups,testing
#SBATCH --nodes=1                                   # node count
#SBATCH --ntasks-per-node=1                         # total number of tasks per node
#SBATCH --cpus-per-task=2		                    # cpu-cores per task (>1 if multi-threaded tasks)
####SBATCH --cpus-per-task=16
#SBATCH --exclusive							
#SBATCH --mem=50G							
#SBATCH --time=2-12						
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

python optimize_property_big.py -snapshot model_snapshot_epoch_200  --hyperparams_path moflow-params.json --batch_size 256 --model_dir results/cancer_v2_128_22   --gpu 0 --max_epochs 3  --weight_decay 1e-3  --data_name cancer  --hidden 100,100,16,  --temperature 1.0  --property_name qed
python optimize_property_big.py -snapshot model_snapshot_epoch_200  --hyperparams_path moflow-params.json --batch_size 256 --model_dir results/cancer_v2_128_22   --gpu 0 --max_epochs 3  --weight_decay 1e-3  --data_name cancer  --hidden 100,100,16,  --temperature 1.0  --property_name plogp


# last default print
echo 'Job done.'
echo -n 'Date: '
date
