import os
import numpy as np
import argparse
from cmdstanpy import CmdStanModel
from datetime import datetime

parser = argparse.ArgumentParser(description="index for the mouse id")
parser.add_argument('mouse_idx', type=str, help='index for the mouse id, starting from 0')
args = parser.parse_args()
mouse_idx = int(args.mouse_idx)

model_path = "/jukebox/witten/yoel/PRE_VIS_DA_CURR_BIO/pipeline/base/"
data_path = "/jukebox/witten/yoel/saved_results/"
save_path = "/jukebox/witten/yoel/saved_results/"

mouse_ids = [13, 14, 15, 16, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43]
mouse = mouse_ids[mouse_idx]
model = CmdStanModel(stan_file=os.path.join(model_path, "beh_model.stan"))

data = np.load(os.path.join(data_path, f"behavior_data_mouseid_{mouse}.npy"))

data_for_stan = {
    "X": data["X"],
    "y": data["y"].astype(int),
    "N": len(data["y"]),
    "P": data["X"].shape[1],
    "S": np.max(data["sesmap"]).item(),
    "NS": data["trial_per_session"].astype(int),
    "ST": data["session_start"].astype(int),
    "SE": data["session_end"].astype(int),
    "sesmap": data["sesmap"].astype(int),
    "choice": data["choice"].astype(int),
}

fit = model.sample(data=data_for_stan, chains=4, iter_warmup=1000, iter_sampling=2000)

save_var_names  = ["betas", "nu", "sigma", "eta", "alpha", "coh_alpha", "L_Sigma"]
unique_dir_name = f"mouseid_{mouse}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
unique_save_path = os.path.join(save_path, unique_dir_name)
os.makedirs(unique_save_path, exist_ok=True)

for svn in save_var_names:
    model_var = fit.stan_variable(svn)
    np.save(os.path.join(unique_save_path, f"{svn}_mouseid_{mouse}.npy"), model_var)
