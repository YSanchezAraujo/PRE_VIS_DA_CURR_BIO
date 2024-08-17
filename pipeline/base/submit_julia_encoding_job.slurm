#!/bin/bash
#SBATCH --job-name=mouse_encoding_fit    # job name
#SBATCH --output=logs_encoding/mouse_encoding_%A_%a.out   # standard output and error log
#SBATCH --error=logs_encoding/mouse_encoding_%A_%a.err    # error log
#SBATCH --array=1-22                  # array range, adjust based on the number of mouse IDs
#SBATCH --time=05:30:00               # time limit hrs:min:sec
#SBATCH --ntasks=1                    # number of tasks (usually 1)
#SBATCH --cpus-per-task=4             # number of CPU cores per task
#SBATCH --mem=35G                      # memory per node
#SBATCH --mail-type=END,FAIL          # mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=sanchez.araujo@princeton.edu  # where to send mail

conda activate /scratch/yaraujjo/stan_env

# define the path to your Python script
JULIA_SCRIPT="/jukebox/witten/yoel/PRE_VIS_DA_CURR_BIO/pipeline/base/run_encoding_model_single_mouse.jl"

# run the Python script with the current SLURM_ARRAY_TASK_ID as the mouse index
julia --project=/jukebox/witten/yoel/PRE_VIS_DA_CURR_BIO/pipeline $JULIA_SCRIPT $SLURM_ARRAY_TASK_ID
