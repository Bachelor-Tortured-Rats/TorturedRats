#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J finetune_wrt_labelsproportion
### -- ask for number of cores (default: 1) --
#BSUB -n 4
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 12:00
# request 32GB of system-memory
#BSUB -R "rusage[mem=64GB]"
#BSUB -R "select[gpu32gb]"
### -- set the email address --
#BSUB -u s204159@student.dtu.dk
### -- send notification at start --
### #BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o finetune_wrt_labelsproportion%J.out
#BSUB -e finetune_wrt_labelsproportion%J.err
# -- end of LSF options --

nvidia-smi
### Load the cuda module
module load python3/3.7.14
module load cuda/10.2
module load cudnn/v8.3.2.44-prod-cuda-10.2
module load ffmpeg/4.2.2


source /zhome/a2/4/155672/Desktop/PythonEnvironments/venv_bachelor/bin/activate
python3 /zhome/a2/4/155672/Desktop/Bachelor/TorturedRats/src/pipelines/finetune_wrt_labelsproportion.py -d hepatic --model_load_path "models/IRCAD__e300_k3_d0.1_lr1E-03_aTrue_bmm.pth"  -a -l "online" -lp '[.1,.2,.3,.5,1]' -e "6"