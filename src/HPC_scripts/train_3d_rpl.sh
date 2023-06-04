#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J train_3drpl_pretask
### -- ask for number of cores (default: 1) --
#BSUB -n 1
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 24:00
# request 32GB of system-memory
#BSUB -R "rusage[mem=256GB]"
#BSUB -R "select[gpu32gb]"
### -- set the email address --
#BSUB -u s204159@student.dtu.dk
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o train_3drpl%J.out
#BSUB -e train_3drpl%J.err
# -- end of LSF options --

nvidia-smi
### Load the cuda module
module load python3/3.7.14
module load cuda/10.2
module load cudnn/v8.3.2.44-prod-cuda-10.2
module load ffmpeg/4.2.2


source $HOME/Desktop/work3S204159/PythonEnvironments/bachelor37/bin/activate
# rat
python3 src/pipelines/train_3drpl.py --num_samples 6 -d rat_kidney --model_save_path "models/selfsupervised_pretask_models/rat_data_final" -lr "1e-4" -e "3000" -l "online"

# hepatic
#python3 src/pipelines/train_3drpl.py -d hepatic --model_save_path "models/selfsupervised_pretask_models/hepatic/final" -lr "1e-5" -e "1500" -l "online"