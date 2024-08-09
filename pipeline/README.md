# Set up compute enviornment in Julia
1. Install Julia from https://julialang.org/downloads/
2. Clone this directory and cd into it
3. start up julia, then press the key with ]
4. run: activate .
5. run: instantiate
6. to get out of the package manager just hit backspace / delete
7. now you should have all packages to run the files in the base directory

# Set up STAN
Ideally you have access to a computing cluster and you ask an admin to install STAN on the cluster, then just pip install cmdstan. If you don't have a computing cluster you will need to install STAN and one of the interfaces to use it (R, Python, Julia, Matlab) see: https://mc-stan.org/users/interfaces/ 

# Alternative to STAN
If you can't install STAN for whatever reason but were successful in installing Julia, I will soon upload an equivalent behavioral model in Julia to the one we used for this paper. This will use the Turing package, though it will likely be slower to run as STAN is typically faster for hierarchical mdoels. 
