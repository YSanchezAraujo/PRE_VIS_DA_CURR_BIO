include(joinpath(@__DIR__, "../base/preprocess.jl"));
include(joinpath(@__DIR__, "../base/constants.jl"));
include(joinpath(@__DIR__, "../base/design_matrix.jl"));
include(joinpath(@__DIR__, "../base/correlation_functions.jl"));
include(joinpath(@__DIR__, "../base/utility.jl"));
include(joinpath(@__DIR__, "../base/figboilerplate.jl"));
include(joinpath(@__DIR__, "../base/encoding_model.jl"));

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

avg_contra_trials = nanmean(contra_trials, 1)
avg_ipsi_trials = nanmean(ipsi_trials, 1)

global_min = minimum([minimum(avg_contra_trials), minimum(avg_ipsi_trials)])
global_max = maximum([maximum(avg_contra_trials), maximum(avg_ipsi_trials)])
mc = pyimport("matplotlib.colors")
color_norm = mc.TwoSlopeNorm(vmin=global_min, vmax=global_max, vcenter=0)

fig, ax = plt.subplots()
cax = ax.imshow(avg_contra_trials[:, :, 2]', aspect="auto", cmap="RdYlBu_r", norm=color_norm)
cbar = fig.colorbar(cax, ax=ax)
ax.set_yticks([0, 4, 9, 14, 19, 24])
ax.set_yticklabels([0, 4, 9, 14, 19, 24] .+ 1)
xt = collect(0:49) ./ 49
ax.set_xticks([0, 25, 49])
ax.set_xticklabels([0, round(xt[25]; digits=1), 1])
ax.set_title("Contralateral")
cbar.ax.yaxis.set_ticks([])
plt.savefig("figure_3c_day0_contra_dms.pdf", bbox_inches="tight", transparent=true)


fig, ax = plt.subplots()
cax = ax.imshow(avg_ipsi_trials[:, :, 2]', aspect="auto", cmap="RdYlBu_r", norm=color_norm)
cbar = fig.colorbar(cax, ax=ax)
ax.set_yticks([0, 4, 9, 14, 19, 24])
ax.set_yticklabels([0, 4, 9, 14, 19, 24] .+ 1)
xt = collect(0:49) ./ 49
ax.set_xticks([0, 25, 49])
ax.set_xticklabels([0, round(xt[25]; digits=1), 1])
ax.set_title("Ipsilateral")
cbar.ax.yaxis.set_ticks([1.5, 0.0, -0.5])

plt.savefig("figure_3c_day0_ipsi_dms.pdf", bbox_inches="tight", transparent=true)
