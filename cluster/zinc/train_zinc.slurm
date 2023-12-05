#!/bin/bash
### Owner: Davide Rigoni

### ============ SLURM COMMANDS
#SBATCH --job-name='MoFlow-train-zinc'
#SBATCH --mail-user=drigoni@math.unipd.it 		## user's email
#SBATCH --error=MoFlow-train-zinc.err
#SBATCH --output=MoFlow-train-zinc.out
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

python train_model.py  --data_name zinc250k  --batch_size  256  --max_epochs 200 --gpu 0  --debug True  --save_dir=results/zinc250k_512t2cnn_256gnn_512-64lin_10flow_19fold_convlu2_38af-1-1mask   --b_n_flow 10  --b_hidden_ch 512,512  --a_n_flow 38  --a_hidden_gnn 256  --a_hidden_lin  512,64   --mask_row_size_list 1 --mask_row_stride_list 1  --noise_scale 0.6  --b_conv_lu 2

# last default print
echo 'Job done.'
echo -n 'Date: '
date
