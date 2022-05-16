#!/bin/bash
### Owner: Davide Rigoni

### ============ SLURM COMMANDS
#SBATCH --job-name='MoFlow-test-zinc-reconstruction'
#SBATCH --mail-user=drigoni@math.unipd.it 		## user's email
#SBATCH --error=MoFlow-test-zinc-reconstruction.err
#SBATCH --output=MoFlow-test-zinc-reconstruction.out
#SBATCH --partition=testing
#SBATCH --ntasks=3
#SBATCH --exclusive								### cores
#SBATCH --mem=50G								### RAM
#SBATCH --time=15-12							### time
#SBATCH --gres=gpu:1							### request GPU with reserved policy
####BATCH --gres=mps:15G							### request GPU with shared policy
####SBATCH --constraint="[dellcuda0|dellsrv1|dellsrv2|dellsrv3]"
#####SBATCH --constraint="[dellcuda0|dellsrv2|dellsrv3|dellsrv4|dellsrv5|dellsrv6]"


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
progPath="${HOME}/repository/moflow/mflow/"
cd $progPath

# input config file for the script
CMD=$1

python generate.py --model_dir results/zinc250k_512t2cnn_256gnn_512-64lin_10flow_19fold_convlu2_38af-1-1mask  -snapshot model_snapshot_epoch_198 --gpu  0  --data_name zinc250k --hyperparams-path moflow-params.json --batch-size 256  --reconstruct   2>&1 | tee zinc250k_reconstruct_results.txt

# last default print
echo 'Job done.'
echo -n 'Date: '
date
