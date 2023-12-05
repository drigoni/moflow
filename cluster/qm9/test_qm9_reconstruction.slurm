#!/bin/bash
### Owner: Davide Rigoni

### ============ SLURM COMMANDS
#SBATCH --job-name='MoFlow-test-qm9-reconstruction'
#SBATCH --mail-user=drigoni@math.unipd.it 		## user's email
#SBATCH --error=MoFlow-test-qm9-reconstruction.err
#SBATCH --output=MoFlow-test-qm9-reconstruction.out
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
progPath="${HOME}/repository/ProgettoCancro-moflow/mflow/"
cd $progPath

# input config file for the script
CMD=$1

python generate.py --model_dir results/qm9_64gnn_128-64lin_1-1mask_0d6noise_convlu1 -snapshot model_snapshot_epoch_200 --gpu 0 --data_name qm9  --hyperparams-path moflow-params.json --batch-size 256 --reconstruct


# last default print
echo 'Job done.'
echo -n 'Date: '
date
