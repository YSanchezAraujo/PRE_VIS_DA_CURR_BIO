#!/bin/bash
#SBATCH --job-name=mouse_model_fit    # Job name
#SBATCH --output=logs/mouse_fit_%A_%a.out   # Standard output and error log
#SBATCH --error=logs/mouse_fit_%A_%a.err    # Error log
#SBATCH --array=0-21                  # Array range, adjust based on the number of mouse IDs
#SBATCH --time=02:00:00               # Time limit hrs:min:sec
#SBATCH --partition=normal            # Partition (queue) name
#SBATCH --ntasks=1                    # Number of tasks (usually 1)
#SBATCH --cpus-per-task=4             # Number of CPU cores per task
#SBATCH --mem=4G                      # Memory per node
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=your.email@domain.com  # Where to send mail

conda activate /scratch/yaraujjo/stan_env


# Define the path to your Python script
PYTHON_SCRIPT="/usr/people/yaraujjo/run_behavioral_model_single_mouse.py"

# Run the Python script with the current SLURM_ARRAY_TASK_ID as the mouse index
python $PYTHON_SCRIPT $SLURM_ARRAY_TASK_ID