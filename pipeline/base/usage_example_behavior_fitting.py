import os
import numpy as np
from cmdstanpy import CmdStanModel

# example usage for single animal
data_path = "/jukebox/witten/yoel/saved_results"

mouse_id = 26
data = np.load(os.path.join(data_path, f"behavior_data_mouseid_{mouse_id}.npy"))

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

model_path = "/jukebox/witten/yoel/PRE_VIS_DA_CURR_BIO/pipeline/base"
model = CmdStanModel(stan_file=os.path.join(model_path, "beh_model.stan"))

fit = model.sample(data=data_for_stan, chains=1, iter_warmup=1000, iter_sampling=1000)
