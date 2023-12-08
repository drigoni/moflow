#!/bin/bash
### Owner: Davide Rigoni

### ============ SLURM COMMANDS
#SBATCH --job-name='MoFlow_interpolation'     # create a short name for your job
#SBATCH --mail-user=davide.rigoni.1@unipd.it
#SBATCH --account=d2023d02-042-users
#SBATCH --output=cluster/out/MoFlow_cancer_v2_interpolation.out
#SBATCH --partition=gpu
#SBATCH --mem=32G				                    # total memory per node
#SBATCH --time=0-05:00:00                           # total run time limit (DD-HH:MM:SS)
#SBATCH --nodes=1                                   # node count
#SBATCH --ntasks-per-node=1                         # total number of tasks per node
#SBATCH --cpus-per-task=16		                    # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gpus-per-node=0                           # number of gpus per node


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

srun python generate.py --model_dir results/cancer26  -snapshot model_snapshot_epoch_200 --gpu  0  --data_name cancer --hyperparams-path moflow-params.json --batch-size 1000  --temperature 0.8 --delta 10  --n_experiments 0 --correct_validity true   --intgrid --inter_times 5

# last default print
echo 'Job done.'
echo -n 'Date: '
date
