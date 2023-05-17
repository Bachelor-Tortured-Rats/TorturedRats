#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J finetune-kfold
### -- ask for number of cores (default: 1) --
#BSUB -n 1
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 10:00
# request 32GB of system-memory
#BSUB -R "rusage[mem=16GB]"
#BSUB -R "select[gpu32gb]"
### -- set the email address --
#BSUB -u s204159@student.dtu.dk
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o HPC_%J.out
#BSUB -e HPC_%J.err
# -- end of LSF options --

nvidia-smi
### Load the cuda module
module load python3/3.7.14
module load cuda/10.2
module load cudnn/v8.3.2.44-prod-cuda-10.2
module load ffmpeg/4.2.2

### activates environment
source /zhome/a2/4/155672/Desktop/PythonEnvironments/venv_bachelor/bin/activate

### folds are 0 to 4

### 3drpl 
### python3 src/pipelines/finetune-kfold.py --k_fold 4 --label_proportion 0.1 --experiement_name "first_exp" --jobid $LSB_JOBID --data_type hepatic  --model_load_path "models/selfsupervised_pretask_models/hepatic/3drpl/3drpl_hepatic__e500_k3_d0_lr1E-04_a_bmm.pth" --setup "3drpl"  --terminate_at_step "24000" --eval_each_steps "200"  --encoder_lr 0  --increase_encoder_lr --learning_rate 1e-4 --wandb_logging "online" 

### random
### python3 src/pipelines/finetune-kfold.py --k_fold 4 --label_proportion 1 --experiement_name "first_exp_random" --jobid $LSB_JOBID --data_type hepatic  --model_load_path "" --setup "random"  --terminate_at_step "24000" --eval_each_steps "200" --encoder_lr 1e-4 --learning_rate 1e-4 --wandb_logging "online" 

### transfer
### python3 src/pipelines/finetune-kfold.py --k_fold 4 --label_proportion .01 --experiement_name "transfer_inc_enc_lr" --jobid $LSB_JOBID --data_type hepatic  --model_load_path "models/IRCAD__e300_k3_d0.1_lr1E-03_aTrue_bmm.pth" --setup "transfer"  --terminate_at_step "24000" --eval_each_steps "200"  --encoder_lr 0  --increase_encoder_lr --learning_rate 1e-4 --wandb_logging "online" 