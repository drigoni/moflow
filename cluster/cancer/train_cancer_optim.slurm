#!/bin/bash
### Owner: Davide Rigoni

### ============ SLURM COMMANDS
#SBATCH --job-name='MoFlow-train-cancer-optim'
#SBATCH --mail-user=drigoni@math.unipd.it 		## user's email
#SBATCH --error=MoFlow-train-cancer-optim.err
#SBATCH --output=MoFlow-train-cancer-optim.out
#SBATCH --partition=testing
#SBATCH --ntasks=3
#SBATCH --exclusive								### cores
#SBATCH --mem=50G								### RAM
#SBATCH --time=15-12							### time
#SBATCH --gres=gpu:1							### request GPU with reserved policy


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

python optimize_property_big.py -snapshot model_snapshot_epoch_200  --hyperparams_path moflow-params.json --batch_size 256 --model_dir results/cancer26   --gpu 0 --max_epochs 3  --weight_decay 1e-3  --data_name cancer  --hidden 100,100,16,  --temperature 1.0  --property_name qed
python optimize_property_big.py -snapshot model_snapshot_epoch_200  --hyperparams_path moflow-params.json --batch_size 256 --model_dir results/cancer26   --gpu 0 --max_epochs 3  --weight_decay 1e-3  --data_name cancer  --hidden 100,100,16,  --temperature 1.0  --property_name plogp


# last default print
echo 'Job done.'
echo -n 'Date: '
date
