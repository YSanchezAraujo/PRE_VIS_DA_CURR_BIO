# Set up compute enviornment in Julia
1. Install Julia from https://julialang.org/downloads/
2. Clone the repo and cd into this directory (because this is where the `.toml` files are)
3. start up julia, then press the key with ]
4. run: `activate .`
5. run: `instantiate`
6. to get out of the package manager just hit backspace / delete
7. now you should have all packages to run the files in the base directory

**Note** The instructions above assume you will be running scripts interactively, if instead you want to run julia from the command line, and want to use the project you can use (for example): 

`julia --project=/path/to/project run_encoding_model.jl`

where `/path/to/project` is the directory where the `.toml` files live.

**Note** Interactive useage: each time you want to run code interactively you will need to cd into the dictory with the `.toml` files, and follow steps 3, 4,and 6 above. 

# Set up STAN
If you have conda installed, a local install of cmdstanpy should be enough: 

This will createa a new environment (recommended, but take note of your system's storage!)

`conda create --prefix /path/to/install -c conda-forge cmdstanpy`

if instead you want to install it into an already established enviornment, remove the prefix: 

`conda install -c conda-forge cmdstanpy`

# Python
Note that the command above will install a new conda environment with it's own version of python, so if you didn't have it installed (unlikely) you now do. 

# Running files
If you are running the julia files in a given directory, open up julia in that directory (or change into it from within julia). So if running for example `run_encoding_model.jl` you should be in the `base` directory when you start up julia. If you are making any of the figures, then `cd` into the `figures` directory before you start up julia

# Reproducing results
After you have set up your compute enviornment and downloaded the data, you should:
1. run `run_encoding_model.jl`
2. run the pre-exposure analysis `day0_encoding_model.jl` and `day0_trial_data.jl`
3. run `save_behavioral_data_for_model.jl`
4. use STAN to run the behavioral model, see `run_behavioral_model_single_mouse.py` and `submit_stan_job.slurm` files
5. aggreate behavioral and neural results, run `aggregate_beh_results.jl` and `aggregate_encoding_results.jl`
    a. (if you ran the encoding model serially and saved the larger results file then no need to aggregrate it)
7. the scripts in the `figures` reproduce figures and results from the paper
