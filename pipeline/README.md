# Set up compute enviornment in Julia
1. Install Julia from https://julialang.org/downloads/
2. Clone the repo and cd into this directory
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

`conda create --prefix /path/to/install -c conda-forge cmdstanpy`

# Reproducing results
After you have set up your compute enviornment and downloaded the data, you should:
1. run `run_encoding_model.jl`
2. run `save_behavioral_data_for_model.jl`
3. use STAN to run the behavioral model
4. aggregate and save results (script to appear)
