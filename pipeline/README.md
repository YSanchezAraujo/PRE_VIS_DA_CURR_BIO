# Set up compute enviornment in Julia
1. Install Julia from https://julialang.org/downloads/
2. Clone this directory and cd into it
3. start up julia, then press the key with ]
4. run: `activate .`
5. run: `instantiate`
6. to get out of the package manager just hit backspace / delete
7. now you should have all packages to run the files in the base directory

# Set up STAN
Ideally you have access to a computing cluster and you ask an admin to install STAN on the cluster, then just pip install cmdstan. If you don't have a computing cluster you will need to install STAN and one of the interfaces to use it (R, Python, Julia, Matlab) see: https://mc-stan.org/users/interfaces/ 

If you have conda installed, a local install of cmdstanpy should be enough: 

`conda create --prefix /path/to/install -c conda-forge cmdstanpy`

# Alternative to STAN
If you can't install STAN for whatever reason but were successful in installing Julia, there is an equivalent model that can be run solely with Julia using the Turing package. Please note: **I recommend you try to get STAN up and running, the Turing model may not be as optimized, and thus will take much longer to run**

# Reproducing results
After you have set up your compute enviornment and downloaded the data, you should:
1. run `run_encoding_model.jl`
2. run `save_behavioral_data_for_model.jl`
3. use either STAN or Turing to run the behavioral model
4. Aggregate and save results (script to appear)
