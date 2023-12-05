#!/bin/bash
### Owner: Davide Rigoni

### ============ SLURM COMMANDS
#SBATCH --job-name='MoFlow-trainoptim-cancer'
#SBATCH --mail-user=davide.rigoni.1@unipd.it 		## user's email
#SBATCH --output=./cluster/out/MoFlow-trainoptimm-cancer_v2.out
#SBATCH --partition=testing
#SBATCH --nodes=1                                   # node count
#SBATCH --ntasks-per-node=1                         # total number of tasks per node
#SBATCH --cpus-per-task=2		                    # cpu-cores per task (>1 if multi-threaded tasks)
####SBATCH --cpus-per-task=16
#SBATCH --exclusive							
#SBATCH --mem=50G							
#SBATCH --time=1-12:00:00						
#SBATCH --gres=gpu:1	
###SBATCH --array=0-13	



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

## PROPERTY=(AVERAGE_GI50 AVERAGE_LC50 AVERAGE_IC50 AVERAGE_TGI) 
#NORMALIZATION=(true true true true true true true false false false false false false false)
#HIDDEN=(15000,4096,1024,128 \
#        1024,512,254,128,64,16,8,4 \
#        2048,1024,128,16 \
#        1024,64 \
#        100,100,32 \
#        100,10 \
#        512,512,64 \
#        15000,4096,1024,128 \
#        1024,512,254,128,64,16,8,4 \
#        2048,1024,128,16 \
#        1024,64 \
#        100,100,32 \
#        100,10 \
#        512,512,64)
#
#        
#if [ ${NORMALIZATION[$SLURM_ARRAY_TASK_ID]} == true ]; 
#then
#echo 'Using Property Normalization'
#srun python optimize_cancer_property.py -snapshot model_snapshot_epoch_200  \
#                                        --hyperparams_path moflow-params.json \
#                                        --batch_size 128 \
#                                        --model_dir results/cancer_v2_128_22   \
#                                        --gpu 0 \
#                                        --max_epochs 10  \
#                                        --weight_decay 1e-3  \
#                                        --data_name cancer  \
#                                        --hidden ${HIDDEN[$SLURM_ARRAY_TASK_ID]}  \
#                                        --temperature 1.0  \
#                                        --property_name AVERAGE_GI50 \
#                                        --norm_property
#else
#echo 'Property NOT Normalized'
#srun python optimize_cancer_property.py -snapshot model_snapshot_epoch_200  \
#                                        --hyperparams_path moflow-params.json \
#                                        --batch_size 128 \
#                                        --model_dir results/cancer_v2_128_22   \
#                                        --gpu 0 \
#                                        --max_epochs 10  \
#                                        --weight_decay 1e-3  \
#                                        --data_name cancer  \
#                                        --hidden ${HIDDEN[$SLURM_ARRAY_TASK_ID]}  \
#                                        --temperature 1.0  \
#                                        --property_name AVERAGE_GI50
#fi

# just one experiment
srun python optimize_cancer_property.py -snapshot model_snapshot_epoch_200  \
                                        --hyperparams_path moflow-params.json \
                                        --batch_size 128 \
                                        --model_dir results/cancer_v2_128_22   \
                                        --gpu 0 \
                                        --max_epochs 8  \
                                        --weight_decay 1e-3  \
                                        --data_name cancer  \
                                        --hidden 1024,512,254,128,64,16,8,4  \
                                        --temperature 1.0  \
                                        --property_name AVERAGE_GI50 \
                                        --norm_property \
                                        --model_suffix _normalized
srun python optimize_cancer_property.py -snapshot model_snapshot_epoch_200  \
                                        --hyperparams_path moflow-params.json \
                                        --batch_size 128 \
                                        --model_dir results/cancer_v2_128_22   \
                                        --gpu 0 \
                                        --max_epochs 8  \
                                        --weight_decay 1e-3  \
                                        --data_name cancer  \
                                        --hidden 2048,1024,128,16  \
                                        --temperature 1.0  \
                                        --property_name AVERAGE_GI50 \
                                        --model_suffix _unnormalized

# last default print
echo 'Job done.'
echo -n 'Date: '
date
