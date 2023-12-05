#!/bin/bash
### Owner: Davide Rigoni

### ============ SLURM COMMANDS
#SBATCH --job-name='MoFlow-train-cancer-128'
#SBATCH --mail-user=davide.rigoni.1@unipd.it 		## user's email
#SBATCH --output=./cluster/out/MoFlow-train-cancer_v2_128-%a.out
#SBATCH --partition=allgroups,testing
#SBATCH --nodes=1                                   # node count
#SBATCH --ntasks-per-node=1                         # total number of tasks per node
#SBATCH --cpus-per-task=2		                    # cpu-cores per task (>1 if multi-threaded tasks)
####SBATCH --cpus-per-task=16
#SBATCH --exclusive							
#SBATCH --mem=50G							
#SBATCH --time=2-12						
#SBATCH --gres=gpu:1						
#SBATCH --array=0-29%3
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

# zinc parameters not working. Just give me directly null errors. The sum of the determinant becomes -inf and the propagate as null
#python train_model.py  --data_name cancer  --batch_size  72  --max_epochs 200 --gpu 0  --debug True  --save_dir=results/cancer_512t2cnn_256gnn_512-64lin_10flow_19fold_convlu2_38af-1-1mask   --b_n_flow 10  --b_hidden_ch 512,512  --a_n_flow 38  --a_hidden_gnn 256  --a_hidden_lin  512,64   --mask_row_size_list 1 --mask_row_stride_list 1  --noise_scale 0.6  --b_conv_lu 2
# qm9 parameters
#python train_model.py  --data_name cancer  --batch_size  216  --max_epochs 200 --gpu 0  --debug True  --save_dir=results/cancer_64gnn_128-64lin_1-1mask_0d6noise_convlu1   --b_n_flow 10  --b_hidden_ch 128,128  --a_n_flow 27 --a_hidden_gnn 64  --a_hidden_lin 128,64  --mask_row_size_list 1 --mask_row_stride_list 1 --noise_scale 0.6 --b_conv_lu 1 --num_workers 1

# version1 NAN schifo, troppi pochi flows da subito
#python train_model.py  --data_name cancer  --batch_size  216  --max_epochs 200 --gpu 0  --debug True  --save_dir=results/cancer1 --b_n_flow 10  --b_hidden_ch 128,128  --a_n_flow 30 --a_hidden_gnn 64  --a_hidden_lin 128,64  --mask_row_size_list 1 --mask_row_stride_list 1 --noise_scale 0.6 --b_conv_lu 2 --num_workers 1
##### version2 NAN ad epoca 107. Unique 70
#python train_model.py   --data_name cancer  --batch_size  216  --max_epochs 200 --gpu 0  --debug True  --save_dir=results/cancer2 --b_n_flow 10  --b_hidden_ch 128,128  --a_n_flow 70 --a_hidden_gnn 64  --a_hidden_lin 128,64  --mask_row_size_list 1 --mask_row_stride_list 1 --noise_scale 0.6 --b_conv_lu 2 --num_workers 1
# version3 NAN schifo
#python train_model.py  --data_name cancer  --batch_size  216  --max_epochs 200 --gpu 0  --debug True  --save_dir=results/cancer3 --b_n_flow 10  --b_hidden_ch 128,128  --a_n_flow 80 --a_hidden_gnn 64  --a_hidden_lin 128,64  --mask_row_size_list 1 --mask_row_stride_list 1 --noise_scale 0.6 --b_conv_lu 2 --num_workers 1
# version4 NAN
#python train_model.py  --data_name cancer  --batch_size  216  --max_epochs 200 --gpu 0  --debug True  --save_dir=results/cancer4 --b_n_flow 10  --b_hidden_ch 128,128  --a_n_flow 100 --a_hidden_gnn 64  --a_hidden_lin 128,64  --mask_row_size_list 1 --mask_row_stride_list 1 --noise_scale 0.6 --b_conv_lu 2 --num_workers 1
# version5 NAN schifo, troppi pochi flows da subito
#python train_model.py  --data_name cancer  --batch_size  216  --max_epochs 200 --gpu 0  --debug True  --save_dir=results/cancer5 --b_n_flow 7  --b_hidden_ch 128,128  --a_n_flow 30 --a_hidden_gnn 64  --a_hidden_lin 128,64  --mask_row_size_list 1 --mask_row_stride_list 1 --noise_scale 0.6 --b_conv_lu 2 --num_workers 1
# version6 NAN
#python train_model.py  --data_name cancer  --batch_size  216  --max_epochs 200 --gpu 0  --debug True  --save_dir=results/cancer6 --b_n_flow 7  --b_hidden_ch 128,128  --a_n_flow 70 --a_hidden_gnn 64  --a_hidden_lin 128,64  --mask_row_size_list 1 --mask_row_stride_list 1 --noise_scale 0.6 --b_conv_lu 2 --num_workers 1
# version7 NAN epoch 61 with uniqueness 63
#python train_model.py  --data_name cancer  --batch_size  216  --max_epochs 200 --gpu 0  --debug True  --save_dir=results/cancer7 --b_n_flow 7  --b_hidden_ch 128,128  --a_n_flow 80 --a_hidden_gnn 64  --a_hidden_lin 128,64  --mask_row_size_list 1 --mask_row_stride_list 1 --noise_scale 0.6 --b_conv_lu 2 --num_workers 1
# version8 NAN da epoca 33. Unique 58
#python train_model.py  --data_name cancer  --batch_size  216  --max_epochs 200 --gpu 0  --debug True  --save_dir=results/cancer8 --b_n_flow 7  --b_hidden_ch 128,128  --a_n_flow 100 --a_hidden_gnn 64  --a_hidden_lin 128,64  --mask_row_size_list 1 --mask_row_stride_list 1 --noise_scale 0.6 --b_conv_lu 2 --num_workers 1
# version9 NAN da subito praticamente
#python train_model.py  --data_name cancer  --batch_size  216  --max_epochs 200 --gpu 0  --debug True  --save_dir=results/cancer9 --b_n_flow 7  --b_hidden_ch 256,256  --a_n_flow 70 --a_hidden_gnn 128  --a_hidden_lin 256,128  --mask_row_size_list 1 --mask_row_stride_list 1 --noise_scale 0.6 --b_conv_lu 2 --num_workers 1
# la versione migliore sembra essere la version2 e la versione 6

# versione10 NAN con uniqueness 65 epoch 37
#python train_model.py   --data_name cancer  --batch_size  216  --max_epochs 200 --gpu 0  --debug True  --save_dir=results/cancer10 --b_n_flow 20  --b_hidden_ch 64,64  --a_n_flow 70 --a_hidden_gnn 32  --a_hidden_lin 64,32  --mask_row_size_list 1 --mask_row_stride_list 1 --noise_scale 0.6 --b_conv_lu 2 --num_workers 1
# versione11 NAN epocs 62 uniq 66
#python train_model.py   --data_name cancer  --batch_size  216  --max_epochs 200 --gpu 0  --debug True  --save_dir=results/cancer11 --b_n_flow 30  --b_hidden_ch 32,32  --a_n_flow 70 --a_hidden_gnn 32  --a_hidden_lin 32,32  --mask_row_size_list 1 --mask_row_stride_list 1 --noise_scale 0.6 --b_conv_lu 2 --num_workers 1
# versione12 NAN
#python train_model.py   --data_name cancer  --batch_size  512  --max_epochs 200 --gpu 0  --debug True  --save_dir=results/cancer12 --b_n_flow 30  --b_hidden_ch 32,32  --a_n_flow 70 --a_hidden_gnn 32  --a_hidden_lin 32,32  --mask_row_size_list 1 --mask_row_stride_list 1 --noise_scale 0.6 --b_conv_lu 2 --num_workers 1
###### versione13 NAN epocs 31 with uniqueness 96
#python train_model.py   --data_name cancer  --batch_size  216  --max_epochs 200 --gpu 0  --debug True  --save_dir=results/cancer13 --b_n_flow 10  --b_hidden_ch 128,128  --a_n_flow 70 --a_hidden_gnn 32  --a_hidden_lin 128,64  --mask_row_size_list 1 --mask_row_stride_list 1 --noise_scale 0.6 --b_conv_lu 2 --num_workers 1
# versione14 GOOD but uniqueness 66
#python train_model.py  --data_name cancer  --batch_size  216  --max_epochs 200 --gpu 0  --debug True  --save_dir=results/cancer14 --b_n_flow 5  --b_hidden_ch 128,128  --a_n_flow 100 --a_hidden_gnn 64  --a_hidden_lin 128,64  --mask_row_size_list 1 --mask_row_stride_list 1 --noise_scale 0.6 --b_conv_lu 2 --num_workers 1

####### versione15 same versione of 13 but with squeeze 35. da qualche segnale di due cicli aromatici, ma molto poco tipo 30
#python train_model.py   --data_name cancer  --batch_size  216  --max_epochs 200 --gpu 0  --debug True  --save_dir=results/cancer15 --b_n_flow 10  --b_hidden_ch 128,128  --a_n_flow 70 --a_hidden_gnn 32  --a_hidden_lin 128,64  --mask_row_size_list 1 --mask_row_stride_list 1 --noise_scale 0.6 --b_conv_lu 2 --num_workers 1
# version16 uguale alla versione zinc. NAN da subito
#python train_model.py  --data_name cancer  --batch_size  216  --max_epochs 200 --gpu 0  --debug True  --save_dir=results/cancer16   --b_n_flow 10  --b_hidden_ch 512,512  --a_n_flow 38  --a_hidden_gnn 256  --a_hidden_lin  512,64   --mask_row_size_list 1 --mask_row_stride_list 1  --noise_scale 0.6  --b_conv_lu 2
# version17 NAN ad epoca 15
#python train_model.py  --data_name cancer  --batch_size  216  --max_epochs 200 --gpu 0  --debug True  --save_dir=results/cancer17   --b_n_flow 10  --b_hidden_ch 512,512  --a_n_flow 70  --a_hidden_gnn 256  --a_hidden_lin  512,64   --mask_row_size_list 1 --mask_row_stride_list 1  --noise_scale 0.6  --b_conv_lu 2
# version18 NAN da subito
#python train_model.py  --data_name cancer  --batch_size  216  --max_epochs 200 --gpu 0  --debug True  --save_dir=results/cancer18   --b_n_flow 20  --b_hidden_ch 512,512  --a_n_flow 70  --a_hidden_gnn 256  --a_hidden_lin  512,64   --mask_row_size_list 1 --mask_row_stride_list 1  --noise_scale 0.6  --b_conv_lu 2

# versione19 Non da segnale di cicli
#python train_model.py   --data_name cancer  --batch_size  216  --max_epochs 200 --gpu 0  --debug True  --save_dir=results/cancer19 --b_n_flow 10  --b_hidden_ch 128,128  --a_n_flow 70 --a_hidden_gnn 64  --a_hidden_lin 128,64  --mask_row_size_list 1 --mask_row_stride_list 1 --noise_scale 0.6 --b_conv_lu 2 --num_workers 1
# versione20 Non da segnale di cicli
#python train_model.py   --data_name cancer  --batch_size  216  --max_epochs 200 --gpu 0  --debug True  --save_dir=results/cancer20 --b_n_flow 5  --b_hidden_ch 128,128  --a_n_flow 70 --a_hidden_gnn 64  --a_hidden_lin 128,64  --mask_row_size_list 1 --mask_row_stride_list 1 --noise_scale 0.6 --b_conv_lu 2 --num_workers 1
# versione21 Da segnali di cicli anche con c3
#python train_model.py   --data_name cancer  --batch_size  216  --max_epochs 200 --gpu 0  --debug True  --save_dir=results/cancer21 --b_n_flow 15  --b_hidden_ch 64,64  --a_n_flow 70 --a_hidden_gnn 64  --a_hidden_lin 64,64  --mask_row_size_list 1 --mask_row_stride_list 1 --noise_scale 0.6 --b_conv_lu 2 --num_workers 1
# versione22 Pochi cicli
#python train_model.py   --data_name cancer  --batch_size  216  --max_epochs 200 --gpu 0  --debug True  --save_dir=results/cancer22 --b_n_flow 10  --b_hidden_ch 128,128  --a_n_flow 70 --a_hidden_gnn 64  --a_hidden_lin 128,64  --mask_row_size_list 1 --mask_row_stride_list 1 --noise_scale 0.6 --b_conv_lu 2 --num_workers 1
# versione23 Pochi cicli, meno della versione 22
#python train_model.py   --data_name cancer  --batch_size  216  --max_epochs 200 --gpu 0  --debug True  --save_dir=results/cancer23 --b_n_flow 10  --b_hidden_ch 128,128  --a_n_flow 38 --a_hidden_gnn 64  --a_hidden_lin 128,64  --mask_row_size_list 1 --mask_row_stride_list 1 --noise_scale 0.6 --b_conv_lu 2 --num_workers 1
# versione24 Un po' peggio della versione 21
#python train_model.py   --data_name cancer  --batch_size  216  --max_epochs 200 --gpu 0  --debug True  --save_dir=results/cancer24 --b_n_flow 10  --b_hidden_ch 128,128  --a_n_flow 38 --a_hidden_gnn 32  --a_hidden_lin 128,64  --mask_row_size_list 1 --mask_row_stride_list 1 --noise_scale 0.6 --b_conv_lu 2 --num_workers 1
######## versione25 come versione 21 ma squeeze 14
#python train_model.py   --data_name cancer  --batch_size  216  --max_epochs 200 --gpu 0  --debug True  --save_dir=results/cancer25 --b_n_flow 15  --b_hidden_ch 64,64  --a_n_flow 70 --a_hidden_gnn 64  --a_hidden_lin 64,64  --mask_row_size_list 1 --mask_row_stride_list 1 --noise_scale 0.6 --b_conv_lu 2 --num_workers 1


########## versione 26 qUESTA VERSIONE HA PIU' CICLI DELLE SUCCESSIVE.
#python train_model.py   --data_name cancer  --batch_size  216  --max_epochs 200 --gpu 0  --debug True  --save_dir=results/cancer26 --b_n_flow 15  --b_hidden_ch 128,128  --a_n_flow 70 --a_hidden_gnn 128  --a_hidden_lin 128,128  --mask_row_size_list 1 --mask_row_stride_list 1 --noise_scale 0.6 --b_conv_lu 2 --num_workers 1
######## versione 27
#python train_model.py   --data_name cancer  --batch_size  216  --max_epochs 200 --gpu 0  --debug True  --save_dir=results/cancer27 --b_n_flow 20  --b_hidden_ch 64,64  --a_n_flow 70 --a_hidden_gnn 64  --a_hidden_lin 64,64  --mask_row_size_list 1 --mask_row_stride_list 1 --noise_scale 0.6 --b_conv_lu 2 --num_workers 1
######## versione 28
#python train_model.py   --data_name cancer  --batch_size  216  --max_epochs 200 --gpu 0  --debug True  --save_dir=results/cancer28 --b_n_flow 20  --b_hidden_ch 128,128  --a_n_flow 100 --a_hidden_gnn 128  --a_hidden_lin 128,128  --mask_row_size_list 1 --mask_row_stride_list 1 --noise_scale 0.6 --b_conv_lu 2 --num_workers 1
######## versione 29
#python train_model.py   --data_name cancer  --batch_size  216  --max_epochs 200 --gpu 0  --debug True  --save_dir=results/cancer29 --b_n_flow 20  --b_hidden_ch 64,64  --a_n_flow 100 --a_hidden_gnn 64  --a_hidden_lin 64,64  --mask_row_size_list 1 --mask_row_stride_list 1 --noise_scale 0.6 --b_conv_lu 2 --num_workers 1



########## versione30 26 cond dimensione maggiorata
#python train_model.py   --data_name cancer  --batch_size  216  --max_epochs 200 --gpu 0  --debug True  --save_dir=results/cancer30 --b_n_flow 15  --b_hidden_ch 256,256  --a_n_flow 70 --a_hidden_gnn 256  --a_hidden_lin 256,256  --mask_row_size_list 1 --mask_row_stride_list 1 --noise_scale 0.6 --b_conv_lu 2 --num_workers 1
#versione31  NAN
#python train_model.py   --data_name cancer  --batch_size  216  --max_epochs 200 --gpu 0  --debug True  --save_dir=results/cancer31 --b_n_flow 15  --b_hidden_ch 512,512  --a_n_flow 70 --a_hidden_gnn 512  --a_hidden_lin 512,512  --mask_row_size_list 1 --mask_row_stride_list 1 --noise_scale 0.6 --b_conv_lu 2 --num_workers 1
#versione32 
#python train_model.py   --data_name cancer  --batch_size  216  --max_epochs 200 --gpu 0  --debug True  --save_dir=results/cancer32 --b_n_flow 20  --b_hidden_ch 256,256  --a_n_flow 100 --a_hidden_gnn 256  --a_hidden_lin 256,256  --mask_row_size_list 1 --mask_row_stride_list 1 --noise_scale 0.6 --b_conv_lu 2 --num_workers 1
#versione33 NAN
#python train_model.py   --data_name cancer  --batch_size  216  --max_epochs 200 --gpu 0  --debug True  --save_dir=results/cancer33 --b_n_flow 15  --b_hidden_ch 512,512  --a_n_flow 70 --a_hidden_gnn 256  --a_hidden_lin 512,512  --mask_row_size_list 1 --mask_row_stride_list 1 --noise_scale 0.6 --b_conv_lu 2 --num_workers 1

# risultati finali
# La miglior versione che fa piu cicli e non fa nan e' la 26! Ho cancellato tutti gli altri esp per via della memoria richiesta.


# training nuova versione


# best 256 15 128,128 70 128 128,128
BATCH_SIZE=(128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128)                                                # 128, 256, 512
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
                            --save_dir results/cancer_v2_128_$SLURM_ARRAY_TASK_ID \
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
