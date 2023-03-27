#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J parameter_opt
### -- ask for number of cores (default: 1) --
#BSUB -n 1
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 24:00
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
#BSUB -o parameter_opt%J.out
#BSUB -e parameter_opt%J.err
# -- end of LSF options --

nvidia-smi
### Load the cuda module
module load python3/3.7.14
module load cuda/10.2
module load cudnn/v8.3.2.44-prod-cuda-10.2
module load ffmpeg/4.2.2


source /zhome/a2/4/155672/Desktop/PythonEnvironments/venv_bachelor/bin/activate
python3 /zhome/a2/4/155672/Desktop/Bachelor/TorturedRats/src/pipelines/hyperparameter_optimizer_agent.py -id "o42h1kpf"