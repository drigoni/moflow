#!/bin/bash
### Owner: Davide Rigoni

### ============ SLURM COMMANDS
#SBATCH --job-name='MoFlow-test-qm9-optimization'
#SBATCH --mail-user=drigoni@math.unipd.it 		## user's email
#SBATCH --error=MoFlow-test-qm9-optimization.err
#SBATCH --output=MoFlow-test-qm9-optimization.out
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
envPath="${HOME}/Programs/miniconda/envs/moflow/bin"
progPath="${HOME}/repository/ProgettoCancro-moflow/mflow/"
cd $progPath

# input config file for the script
CMD=$1

python optimize_property.py -snapshot model_snapshot_epoch_200  --hyperparams_path moflow-params.json --batch_size 256 --model_dir results/qm9_64gnn_128-64lin_1-1mask_0d6noise_convlu1   --gpu 0   --data_name qm9   --property_name qed --topk 2000  --property_model_path qed_model.pt --debug false  --topscore

# last default print
echo 'Job done.'
echo -n 'Date: '
date
