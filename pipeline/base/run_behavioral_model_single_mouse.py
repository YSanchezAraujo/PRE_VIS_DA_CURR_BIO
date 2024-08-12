import os
import numpy as np
import argparse
from cmdstanpy import CmdStanModel
from datetime import datetime

parser = argparse.ArgumentParser(description="index for the mouse id")
parser.add_argument('mouse_idx', type=str, help='index for the mouse id, starting from 0')
args = parser.parse_args()
mouse_idx = int(args.mouse_idx)

data_path = "/usr/people/yaraujjo/"
save_path = "/usr/people/yaraujjo/beh_model_fits"

mouse_ids = [13, 14, 15, 16, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43]
mouse = mouse_ids[mouse_idx]
model = CmdStanModel(stan_file=os.path.join(data_path, "beh_model.stan"))

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

fit = model.sample(data=data_for_stan, chains=4, iter_warmup=1000, iter_sampling=1000)

# extract betas and compute mean over all samples
betas = fit.stan_variable("betas").mean(0)
# save them for later processing
unique_dir_name = f"betas_mouseid_{mouse}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
unique_save_path = os.path.join(save_path, unique_dir_name)
os.makedirs(unique_save_path, exist_ok=True)
np.save(os.path.join(unique_save_path, f"betas_mouseid_{mouse}.npy"), betas)
print(f"Betas saved in directory: {unique_save_path}")