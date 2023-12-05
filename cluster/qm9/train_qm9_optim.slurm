#!/bin/bash
### Owner: Davide Rigoni

### ============ SLURM COMMANDS
#SBATCH --job-name='MoFlow-train-qm9-optim'
#SBATCH --mail-user=drigoni@math.unipd.it 		## user's email
#SBATCH --error=MoFlow-train-qm9-optim.err
#SBATCH --output=MoFlow-train-qm9-optim.out
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

python optimize_property.py -snapshot model_snapshot_epoch_200  --hyperparams_path moflow-params.json --batch_size 256 --model_dir results/qm9_64gnn_128-64lin_1-1mask_0d6noise_convlu1   --gpu 0 --max_epochs 3  --weight_decay 1e-3  --data_name qm9  --hidden 16,  --temperature 1.0  --property_name qed
python optimize_property.py -snapshot model_snapshot_epoch_200  --hyperparams_path moflow-params.json --batch_size 256 --model_dir results/qm9_64gnn_128-64lin_1-1mask_0d6noise_convlu1   --gpu 0 --max_epochs 3  --weight_decay 1e-3  --data_name qm9  --hidden 16,  --temperature 1.0  --property_name plogp


# last default print
echo 'Job done.'
echo -n 'Date: '
date
