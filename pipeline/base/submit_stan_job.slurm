#!/bin/bash
#SBATCH --job-name=mouse_model_fit    # job name
#SBATCH --output=logs/mouse_fit_%A_%a.out   # standard output and error log
#SBATCH --error=logs/mouse_fit_%A_%a.err    # error log
#SBATCH --array=0-21                  # array range, adjust based on the number of mouse IDs
#SBATCH --time=18:00:00               # time limit hrs:min:sec
#SBATCH --ntasks=1                    # number of tasks (usually 1)
#SBATCH --cpus-per-task=4             # number of CPU cores per task
#SBATCH --mem=4G                      # memory per node
#SBATCH --mail-type=END,FAIL          # mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=sanchez.araujo@princeton.edu  # where to send mail

conda activate /scratch/yaraujjo/stan_env

# define the path to your Python script
PYTHON_SCRIPT="/jukebox/witten/yoel/PRE_VIS_DA_CURR_BIO/pipeline/base/run_behavioral_model_single_mouse.py"

# run the Python script with the current SLURM_ARRAY_TASK_ID as the mouse index
python $PYTHON_SCRIPT $SLURM_ARRAY_TASK_ID
