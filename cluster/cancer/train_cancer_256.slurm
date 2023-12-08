#!/bin/bash
### Owner: Davide Rigoni

### ============ SLURM COMMANDS
#SBATCH --job-name='MoFlow_cancer_train_256'     # create a short name for your job
#SBATCH --mail-user=davide.rigoni.1@unipd.it
#SBATCH --account=d2023d02-042-users
#SBATCH --output=cluster/out/MoFlow_cancer_v3_train-%a.out
#SBATCH --partition=gpu
#SBATCH --mem=128G				                    # total memory per node
#SBATCH --time=2-00:00:00                           # total run time limit (DD-HH:MM:SS)
#SBATCH --nodes=1                                   # node count
#SBATCH --ntasks-per-node=1                         # total number of tasks per node
#SBATCH --cpus-per-task=64		                    # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gpus-per-node=1                           # number of gpus per node				
#SBATCH --array=0-9
####SBATCH --array=0


### ============ VARIABLES SETTING
SLURM_MASTER_PORT=$(expr 10000 + $(echo -n ${SLURM_JOBID} | tail -c 4))
SLURM_MASTER_NODE="$(scontrol show hostnames ${SLURM_JOB_NODELIST} | head -n 1)" # such as gn11
SLURM_MASTER_ADDR="${SLURM_MASTER_NODE}"
SLURM_MASTER_URL="tcp://${SLURM_MASTER_ADDR}:${SLURM_MASTER_PORT}"
export SLURM_MASTER_PORT=$SLURM_MASTER_PORT
export SLURM_MASTER_NODE=$SLURM_MASTER_NODE
export SLURM_MASTER_ADDR=$SLURM_MASTER_ADDR

# removed for this project
# # model variables
# NUM_GPUS=${SLURM_GPUS_PER_NODE}
# NUM_MACHINES=${SLURM_NNODES}
# BATCH_SIZE=$(expr 5 \* $(echo -n ${SLURM_JOB_NUM_NODES}) \* $(echo -n ${SLURM_GPUS_PER_NODE})) # max 5 per A100
# MAX_ITER=$(expr 32 \* 90000 \/ $(echo -n ${BATCH_SIZE}))

### ============ SOME PRINT COMMANDS
echo ''
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo 'Date: ' $(date)
echo 'Directory: ' $(pwd)
echo "Nodelist: " ${SLURM_JOB_NODELIST}                  # like gn11 or gn[37,58]
echo "Number of nodes: " ${SLURM_JOB_NUM_NODES}
echo "Ntasks per node: "  ${SLURM_NTASKS_PER_NODE}
echo "NGPUs per node: "  ${SLURM_GPUS_PER_NODE}
echo "CUDA_VISIBLE_DEVICES: " ${CUDA_VISIBLE_DEVICES}
#echo "TORCH_DEVICE_COUNT: " $(python -c 'import torch; print(torch.cuda.device_count())')
echo "SLURM_MASTER_PORT: " ${SLURM_MASTER_PORT}
echo "SLURM_MASTER_NODE: " ${SLURM_MASTER_NODE}
echo "SLURM_MASTER_ADDR: " ${SLURM_MASTER_ADDR}
echo "SLURM_MASTER_URL: " ${SLURM_MASTER_URL}
# echo "--------------------------------------------- "
# echo "MODEL_NUM_GPUS: " ${NUM_GPUS}
# echo "MODEL_NUM_MACHINES: " ${NUM_MACHINES}
# echo "MODEL_BATCH_SIZE: " ${BATCH_SIZE}
# echo "MODEL_MAX_ITER: " ${MAX_ITER}
echo "--------------------------------------------- "
echo Current GPUs state:
nvidia-smi
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo ''

cd "/ceph/hpc/home/eudavider/repository/ProgettoCancro-moflow/mflow/"


BATCH_SIZE=(256 256 256 256 256 256 256 256 256 256) 
B_N_FLOW=(7 7 20 20 7 7 20 20 10 10)                                                             
B_HIDDEN_CH=('128,128' '256,256' '128,128' '256,256' '128,128' '256,256' '128,128' '256,256' '512,512' '512,512')
A_N_FLOW=(100 100 100 100 130 130 130 130 38 100)                                            
A_HIDDEN_GNN=(128 256 128 256 128 256 128 256 256 256) 
A_HIDDEN_LIN=('128,128' '256,256' '128,128' '256,256' '128,128' '256,256' '128,128' '256,256' '512,64' '512,64')

srun python train_model.py  --data_name cancer  \
                            --batch_size  ${BATCH_SIZE[$SLURM_ARRAY_TASK_ID]}  \
                            --max_epochs 300 \
                            --gpu 0  \
                            --debug False  \
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
                            --num_workers 32 \
                            --save_interval 50


# python train_model.py  --data_name cancer  \
#                             --batch_size  512  \
#                             --max_epochs 2 \
#                             --gpu 0  \
#                             --debug False  \
#                             --save_dir results/prova \
#                             --b_n_flow 7  \
#                             --b_hidden_ch '64,64'  \
#                             --a_n_flow 30 \
#                             --a_hidden_gnn 64  \
#                             --a_hidden_lin '64,64'  \
#                             --mask_row_size_list 1 \
#                             --mask_row_stride_list 1 \
#                             --noise_scale 0.6 \
#                             --b_conv_lu 2 \
#                             --num_workers 32 \
#                             --save_interval 1



# NOTE
# 28 meglio di 29.
# latent space dimension importa.

# last default print
echo '-----------------------------------'
echo 'Job done.'
echo -n 'Date: '
date
