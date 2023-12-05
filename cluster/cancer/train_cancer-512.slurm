#!/bin/bash
### Owner: Davide Rigoni

### ============ SLURM COMMANDS
#SBATCH --job-name='MoFlow-train-cancer-512'
#SBATCH --mail-user=davide.rigoni.1@unipd.it 		## user's email
#SBATCH --output=./cluster/out/MoFlow-train-cancer_v2_512-%a.out
#SBATCH --partition=testing
#SBATCH --nodes=1                                   # node count
#SBATCH --ntasks-per-node=1                         # total number of tasks per node
#SBATCH --cpus-per-task=2		                    # cpu-cores per task (>1 if multi-threaded tasks)
####SBATCH --cpus-per-task=16
#SBATCH --exclusive							
#SBATCH --mem=50G							
#SBATCH --time=2-12						
#SBATCH --gres=gpu:1						
#SBATCH --array=0-29%6
####SBATCH --array=0


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

# best 256 15 128,128 70 128 128,128
BATCH_SIZE=(512 512 512 512 512 512 512 512 512 512 512 512 512 512 512 512 512 512 512 512 512 512 512 512 512 512 512 512 512 512)                                                # 128, 256, 512
B_N_FLOW=(7 7 7 7 7 7 10 10 10 10 10 10 15 15 15 15 15 15 20 20 20 20 20 20 30 30 30 30 30 30)                                                              # 7, 10,15,20,30
B_HIDDEN_CH=('64,64' '128,128' '256,256' '64,64' '128,128' '256,256' '64,64' '128,128' '256,256' '64,64' '128,128' '256,256' '64,64' '128,128' '256,256' '64,64' '128,128' '256,256' '64,64' '128,128' '256,256' '64,64' '128,128' '256,256' '64,64' '128,128' '256,256' '64,64' '128,128' '256,256')               # 64,64 128,128, 256,256
A_N_FLOW=(70 70 70 100 100 100 70 70 70 100 100 100 70 70 70 100 100 100 70 70 70 100 100 100 70 70 70 100 100 100)                                                     # 70, 100
A_HIDDEN_GNN=(64 128 256 64 128 256 64 128 256 64 128 256 64 128 256 64 128 256 64 128 256 64 128 256 64 128 256 64 128 256)                                                # 64, 128, 256
A_HIDDEN_LIN=('64,64' '128,128' '256,256' '64,64' '128,128' '256,256' '64,64' '128,128' '256,256' '64,64' '128,128' '256,256' '64,64' '128,128' '256,256' '64,64' '128,128' '256,256' '64,64' '128,128' '256,256' '64,64' '128,128' '256,256' '64,64' '128,128' '256,256' '64,64' '128,128' '256,256')              # 64,64 128,128 256,256 

# model config
srun python ${FILE_PYTHON[$SLURM_ARRAY_TASK_ID]} ${NAME_FOLDER} ${FILE_MOLECULES[$SLURM_ARRAY_TASK_ID]} ${DATASET} ${NAME_EXPERIMENT}

srun python train_model.py  --data_name cancer  \
                            --batch_size  ${BATCH_SIZE[$SLURM_ARRAY_TASK_ID]}  \
                            --max_epochs 200 \
                            --gpu 0  \
                            --debug True  \
                            --save_dir results/cancer_v2_512_$SLURM_ARRAY_TASK_ID \
                            --b_n_flow ${B_N_FLOW[$SLURM_ARRAY_TASK_ID]}  \
                            --b_hidden_ch ${B_HIDDEN_CH[$SLURM_ARRAY_TASK_ID]}  \
                            --a_n_flow ${A_N_FLOW[$SLURM_ARRAY_TASK_ID]} \
                            --a_hidden_gnn ${A_HIDDEN_GNN[$SLURM_ARRAY_TASK_ID]}  \
                            --a_hidden_lin ${A_HIDDEN_LIN[$SLURM_ARRAY_TASK_ID]}  \
                            --mask_row_size_list 1 \
                            --mask_row_stride_list 1 \
                            --noise_scale 0.6 \
                            --b_conv_lu 2 \
                            --num_workers 2



# NOTE
# 28 meglio di 29.
# latent space dimension importa.

# last default print
echo '-----------------------------------'
echo 'Job done.'
echo -n 'Date: '
date
