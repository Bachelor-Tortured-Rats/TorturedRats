#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J finetune_wrt_labelsproportion_3drpl
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
python3 /zhome/a2/4/155672/Desktop/Bachelor/TorturedRats/src/pipelines/finetune_wrt_labelsproportion.py —-LSB_JOBID $LSB_JOBID -d hepatic -s "3drpl" -l "online" --model_load_path "models/selfsupervised_pretask_models/hepatic/3drpl/3drpl_hepatic__e500_k3_d0_lr1E-04_a_bmm.pth" -lp '[.01,0.03,0.06,0.10,1.00]' --terminate_at_step_count "24000"
