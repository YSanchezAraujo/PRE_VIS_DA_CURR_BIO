include(joinpath(@__DIR__, "preprocess.jl"))
include(joinpath(@__DIR__, "design_matrix.jl"))
include(joinpath(@__DIR__, "constants.jl"))
include(joinpath(@__DIR__, "utility.jl"))



function get_stim_resp_day0(mouse_id, base_path, window, n_trial, region)
    data = mouse_session0_data(base_path, mouse_id);
    
    if region == "DMS"
        z_neu = data.neural.DMS
        beh_contra_map = beh_dms_contra_map
        beh_ipsi_map = beh_dms_ipsi_map
    end

    if region == "DLS"
        z_neu = data.neural.DLS
        beh_contra_map = beh_dms_ipsi_map
        beh_ipsi_map = beh_dms_contra_map
    end

    if region == "NAcc"
        z_neu = data.neural.NAcc
        beh_contra_map = beh_dms_ipsi_map
        beh_ipsi_map = beh_dms_contra_map
    end

    beh_contrast_df = add_intercept([data.behavior.contrast_left data.behavior.contrast_right])
    # contra stim response
    beh_contra_stim = beh_contrast_df[:, beh_contra_map[mouse_id]]
    trial_bool = [beh_contra_stim .== j for j in sort(unique(beh_contra_stim))[1:end-1]] # last one will be nan
    trial_idx = data.stim_idx[trial_bool[4]]
    n_trial_contra = minimum([length(trial_idx), n_trial])
    neu_contra_stim_resp_100 = hcat([z_neu[trial_idx[j]:trial_idx[j]+window-1] for j in 1:n_trial_contra]...)

    # now for ipsi
    beh_ipsi_stim = beh_contrast_df[:, beh_ipsi_map[mouse_id]]
    trial_bool = [beh_ipsi_stim .== j for j in sort(unique(beh_ipsi_stim))[1:end-1]] # last one will be nan
    trial_idx = data.stim_idx[trial_bool[4]]
    n_trial_ipsi = minimum([length(trial_idx), n_trial])
    neu_ipsi_stim_resp_100 = hcat([z_neu[trial_idx[j]:trial_idx[j]+window-1] for j in 1:n_trial_ipsi]...)

    return (
        contra = neu_contra_stim_resp_100,
        ipsi = neu_ipsi_stim_resp_100,
        n_trial_contra = n_trial_contra,
        n_trial_ipsi = n_trial_ipsi
    )
end

base_path = "/jukebox/witten/ONE/alyx.internationalbrainlab.org/wittenlab/Subjects"
window = 50
n_trial = 25
mice_ids = collect(26:43)
n_mice = length(mice_ids)
reg_labs = ["NAcc", "DMS", "DLS"]

contra_trials = fill(NaN, n_mice, window, n_trial, length(reg_labs));
ipsi_trials = fill(NaN, n_mice, window, n_trial, length(reg_labs));


for (i, mouse_id) in enumerate(mice_ids)
    for (j, region) in enumerate(reg_labs)
        d0data = get_stim_resp_day0(mouse_id, base_path, window, n_trial, region)
        contra_trials[i, :, 1:d0data.n_trial_contra, j] = d0data.contra
        ipsi_trials[i, :, 1:d0data.n_trial_ipsi, j] = d0data.ipsi
    end
end

H1 = nanmean(contra_trials, 1)
H2 = nanmean(ipsi_trials, 1)

global_min = minimum([minimum(H1), minimum(H2)])
global_max = maximum([maximum(H1), maximum(H2)])

fig, ax = subplots(nrows=3, ncols=2)
for j in 1:3
    ax[j, 1].imshow(H1[:, :, j]', aspect="auto", cmap="RdYlBu_r", vmin=global_min, vmax=global_max)
    ax[j, 2].imshow(H2[:, :, j]', aspect="auto", cmap="RdYlBu_r", vmin=global_min, vmax=global_max)
end
