

const dms_contra_map = Dict(
    13 => "stim_left",
    14 => "stim_left",
    15 => "stim_left",
    16 => "stim_left",
    26 => "stim_right",
    27 => "stim_right",
    28 => "stim_right",
    29 => "stim_right",
    30 => "stim_left",
    31 => "stim_left",
    32 => "stim_left",
    33 => "stim_left",
    34 => "stim_left",
    35 => "stim_left",
    36 => "stim_left",
    37 => "stim_right",
    38 => "stim_right",
    39 => "stim_left",
    40 => "stim_right",
    41 => "stim_right",
    42 => "stim_left",
    43 => "stim_left", 
)

g_beh_contra_ipsi(x) = x == "stim_left" ? 2 : 3
const beh_dms_contra_map = Dict(
    f => g_beh_contra_ipsi(dms_contra_map[f]) for f in [collect(13:16); collect(26:43)]
)


const dms_ipsi_map = Dict(
    13 => "stim_right",
    14 => "stim_right",
    15 => "stim_right",
    16 => "stim_right",
    26 => "stim_left",
    27 => "stim_left",
    28 => "stim_left",
    29 => "stim_left",
    30 => "stim_right",
    31 => "stim_right",
    32 => "stim_right",
    33 => "stim_right",
    34 => "stim_right",
    35 => "stim_right",
    36 => "stim_right",
    37 => "stim_left",
    38 => "stim_left",
    39 => "stim_right",
    40 => "stim_left",
    41 => "stim_left",
    42 => "stim_right",
    43 => "stim_right", 
)

const beh_dms_ipsi_map = Dict(
    f => g_beh_contra_ipsi(dms_ipsi_map[f]) for f in [collect(13:16); collect(26:43)]
)

g_beh_mult(x) = x == 3 ? -1 : 1
const beh_multiplier_contra = Dict(
    f => g_beh_mult(beh_dms_contra_map[f]) for f in [collect(13:16); collect(26:43)]
)

const beh_multiplier_ipsi = Dict(
    f => g_beh_mult(beh_dms_ipsi_map[f]) for f in [collect(13:16); collect(26:43)]
)

const event_names = [
    "stim_right", "stim_left", 
    "act_right_correct", "act_right_incorrect",
    "act_left_correct", "act_left_incorrect", 
    "reward_right_correct", "reward_right_incorrect", 
    "reward_left_correct", "reward_left_incorrect"
];
