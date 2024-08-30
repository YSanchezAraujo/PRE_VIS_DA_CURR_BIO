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
cax = ax.imshow(avg_contra_trials[:, :, 1]', aspect="auto", cmap="RdYlBu_r", norm=color_norm)
cbar = fig.colorbar(cax, ax=ax)
ax.set_yticks([0, 4, 9, 14, 19, 24])
ax.set_yticklabels([0, 4, 9, 14, 19, 24] .+ 1)
xt = collect(0:49) ./ 49
ax.set_xticks([0, 25, 49])
ax.set_xticklabels([0, round(xt[25]; digits=1), 1])
ax.set_title("Contralateral")
cbar.ax.yaxis.set_ticks([])
plt.savefig("supp_figure_day0_contra_nacc.pdf", bbox_inches="tight", transparent=true)

fig, ax = plt.subplots()
cax = ax.imshow(avg_ipsi_trials[:, :, 1]', aspect="auto", cmap="RdYlBu_r", norm=color_norm)
cbar = fig.colorbar(cax, ax=ax)
ax.set_yticks([0, 4, 9, 14, 19, 24])
ax.set_yticklabels([0, 4, 9, 14, 19, 24] .+ 1)
xt = collect(0:49) ./ 49
ax.set_xticks([0, 25, 49])
ax.set_xticklabels([0, round(xt[25]; digits=1), 1])
ax.set_title("Ipsilateral")
cbar.ax.yaxis.set_ticks([1.5, 0.0, -0.5])
plt.savefig("supp_figure_day0_ipsi_nacc.pdf", bbox_inches="tight", transparent=true)

fig, ax = plt.subplots()
cax = ax.imshow(avg_contra_trials[:, :, 3]', aspect="auto", cmap="RdYlBu_r", norm=color_norm)
cbar = fig.colorbar(cax, ax=ax)
ax.set_yticks([0, 4, 9, 14, 19, 24])
ax.set_yticklabels([0, 4, 9, 14, 19, 24] .+ 1)
xt = collect(0:49) ./ 49
ax.set_xticks([0, 25, 49])
ax.set_xticklabels([0, round(xt[25]; digits=1), 1])
ax.set_title("Contralateral")
cbar.ax.yaxis.set_ticks([])
plt.savefig("supp_figure_day0_contra_dls.pdf", bbox_inches="tight", transparent=true)


fig, ax = plt.subplots()
cax = ax.imshow(avg_ipsi_trials[:, :, 3]', aspect="auto", cmap="RdYlBu_r", norm=color_norm)
cbar = fig.colorbar(cax, ax=ax)
ax.set_yticks([0, 4, 9, 14, 19, 24])
ax.set_yticklabels([0, 4, 9, 14, 19, 24] .+ 1)
xt = collect(0:49) ./ 49
ax.set_xticks([0, 25, 49])
ax.set_xticklabels([0, round(xt[25]; digits=1), 1])
ax.set_title("Ipsilateral")
cbar.ax.yaxis.set_ticks([1.5, 0.0, -0.5])
plt.savefig("supp_figure_day0_ipsi_dls.pdf", bbox_inches="tight", transparent=true)

# now for the other panels and statistics
using AnovaBase;
using AnovaMixedModels;

neu_results = load("/jukebox/witten/yoel/saved_results/neural_results.jld2", "results");
behavior_weight = load("/jukebox/witten/yoel/saved_results/choice_weights.jld2", "results");
kernel_norm = neu_results.kernel_norm;
mouse_ids = [collect(13:16); collect(26:43)];
n_mice = length(mouse_ids);

qc_dict = Dict(
    f => get_qc("/jukebox/witten/ONE/alyx.internationalbrainlab.org/wittenlab/Subjects", f)
    for f in mouse_ids
    )

function make_nan_from_qc(kernel_norm, qc_dict, mouse_ids)
    for mouse in mouse_ids
        for evn in event_names
            for reg in ["NAcc", "DMS", "DLS"]
                qc_days = qc_dict[mouse].train.session
                qc_reg = qc_dict[mouse].train[!, Symbol("QC_$(reg)")]
                make_nan = qc_reg .== 0
                kernel_norm[mouse][reg][evn][qc_days[make_nan], :] .= NaN
            end
        end
    end

    return kernel_norm
end

kernel_norm = make_nan_from_qc(kernel_norm, qc_dict, mouse_ids)
neu_day0_results = load("/jukebox/witten/yoel/saved_results/day0_neural_results.jld2", "results");
day0_mice = 26:43;



# figure 3.D
day0_contra_dls_norms = hcat([neu_day0_results.kernel_norm[f]["DLS"][dms_ipsi_map[f]] for f in day0_mice]...)'
day0_contra_dls_norms_conmod = day0_contra_dls_norms[:, 4] .- day0_contra_dls_norms[:, 1]
strong_dls = day0_contra_dls_norms_conmod .>= median(day0_contra_dls_norms_conmod)
weak_dls = .!strong_dls
strong_color="darkgreen"
weak_color="lightgreen"
bin_width = 0.2  # Example bin width; adjust as necessary
bins_strong = collect(median(day0_contra_dls_norms_conmod):bin_width:maximum(day0_contra_dls_norms_conmod))
bins_weak = collect(minimum(day0_contra_dls_norms_conmod):bin_width:median(day0_contra_dls_norms_conmod))

fig, ax = plt.subplots()
ax.hist(
    day0_contra_dls_norms_conmod[strong_dls], 
    bins=bins_strong, color=strong_color, edgecolor="white", alpha=1, label="Strong"
    )
ax.hist(
    day0_contra_dls_norms_conmod[weak_dls],
     bins=bins_weak, color=weak_color, edgecolor="white", alpha=1, label="Weak"
    )
ax.set_ylabel("Counts")
ax.axvline(median(day0_contra_dls_norms_conmod), lw=2, linestyle="--", color="black", label="_nolegend_")
#plt.legend()
plt.savefig("strong_weak_dls_day0_hist.pdf", transparent=true, bbox_inches="tight")

function get_side_map_multiplier(side_label, region_label)
    if side_label == "contra" && region_label == "DMS"
        return beh_dms_contra_map, 1
    end

    if side_label == "contra" && region_label == "NAcc"
        return beh_dms_ipsi_map, -1
    end

    if side_label == "contra" && region_label == "DLS"
        return beh_dms_ipsi_map, -1
    end

    if side_label == "ipsi" && region_label == "DMS"
        return beh_dms_ipsi_map, -1
    end

    if side_label == "ipsi" && region_label == "NAcc"
        return beh_dms_contra_map, 1
    end

    if side_label == "ipsi" && region_label == "DLS"
        return beh_dms_contra_map, 1
    end
end

function weight_by_group(beh_weight, group_ids, side_label, region_label)
    side_map, multiplier = get_side_map_multiplier(side_label, region_label)

    group_avgs = hcat([
        behavior_weight[f].avg[:, side_map[f]] * (multiplier * beh_multiplier_contra[f]) for f in group_ids
        ]...)

    group_ptl = hcat([
        behavior_weight[f].ci275[:, side_map[f]] * (multiplier * beh_multiplier_contra[f]) for f in group_ids
        ]...)

    group_ptu = hcat([
        behavior_weight[f].ci975[:, side_map[f]] * (multiplier * beh_multiplier_contra[f]) for f in group_ids
        ]...)

    return (
        mice = group_avgs,
        avg = nanmean(group_avgs, 2),
        sem = nansem(group_avgs, 2),
        ci275 = nanmean(group_ptl, 2),
        ci975 = nanmean(group_ptu, 2)
    )

end

strong_grp_ids = collect(26:43)[strong_dls]
weak_grp_ids = collect(26:43)[weak_dls]

beh_avg_strong_dls_contra_day0 = weight_by_group(behavior_weight, strong_grp_ids, "contra", "DLS")
beh_avg_weak_dls_contra_day0 = weight_by_group(behavior_weight, weak_grp_ids, "contra", "DLS")

# figure 3.E
fig, ax = plt.subplots()
xt = 1:20
ax.plot(xt, beh_avg_strong_dls_contra_day0.avg, color=strong_color, alpha=1)
ax.fill_between(xt, beh_avg_strong_dls_contra_day0.avg .- beh_avg_strong_dls_contra_day0.sem, 
                     beh_avg_strong_dls_contra_day0.avg .+ beh_avg_strong_dls_contra_day0.sem, 
                     alpha=0.5, color=strong_color, label="_nolegend_")

ax.plot(xt, beh_avg_weak_dls_contra_day0.avg, color=weak_color)
ax.fill_between(xt, beh_avg_weak_dls_contra_day0.avg .- beh_avg_weak_dls_contra_day0.sem, 
                     beh_avg_weak_dls_contra_day0.avg .+ beh_avg_weak_dls_contra_day0.sem, 
                     alpha=0.5, color=weak_color, label="_nolegend_", )

ax.set_xticks([1, 10, 20])
ax.set_yticks([0, 4, 8])
ax.set_xlabel("Session")
plt.savefig("strong_weak_choice_weights_contra_dls.pdf", bbox_inches="tight", transparent=true)
plt.close()

# check stats 
strong_contra_dls_df = df_for_stats(beh_avg_strong_dls_contra_day0.mice, day0_contra_dls_norms_conmod[strong_dls], strong_grp_ids)
weak_contra_dls_df = df_for_stats(beh_avg_weak_dls_contra_day0.mice, day0_contra_dls_norms_conmod[weak_dls], weak_grp_ids)
contra_df_dls = [strong_contra_dls_df; weak_contra_dls_df]
contra_df_dls[!, :mc_neural_strength] = contra_df_dls.neural_strength .- mean(contra_df_dls.neural_strength);
frm = @formula(contrast_weight ~ day_split3*mc_neural_strength + (1 + day_split3|mouse))
contrasts = Dict(
    :mouse => DummyCoding(), 
    :day_split3 => DummyCoding()
)
contra_model = lme(frm, contra_df_dls, contrasts=contrasts)
anova(contra_model, type=3)

# figure 3.F
beh_avg_strong_dls_ipsi_day0 = weight_by_group(behavior_weight, strong_grp_ids, "ipsi", "DLS")
beh_avg_weak_dls_ipsi_day0 = weight_by_group(behavior_weight, weak_grp_ids, "ipsi", "DLS")

fig, ax = plt.subplots()
ax.plot(xt, beh_avg_strong_dls_ipsi_day0.avg, color=strong_color, label="Strong")
ax.fill_between(xt, beh_avg_strong_dls_ipsi_day0.avg .- beh_avg_strong_dls_ipsi_day0.sem, 
                     beh_avg_strong_dls_ipsi_day0.avg .+ beh_avg_strong_dls_ipsi_day0.sem, 
                     alpha=0.4, color=strong_color, label="_nolegend_")

ax.plot(xt, beh_avg_weak_dls_ipsi_day0.avg, color=weak_color, label="Weak")
ax.fill_between(xt, beh_avg_weak_dls_ipsi_day0.avg .- beh_avg_weak_dls_ipsi_day0.sem, 
                     beh_avg_weak_dls_ipsi_day0.avg .+ beh_avg_weak_dls_ipsi_day0.sem, 
                     alpha=0.4, color=weak_color, label="_nolegend_")

ax.set_xticks([1, 10, 20])
ax.set_yticks([0, 4, 8])
ax.set_xlabel("Session")
plt.savefig("strong_weak_choice_weights_ipsi_dls.pdf", bbox_inches="tight", transparent=true)
plt.close()

# need to make the dataframes and do the stats like i do for all the other figure 3.X
strong_ipsi_df = df_for_stats(beh_avg_strong_dls_ipsi_day0.mice, day0_contra_dls_norms_conmod[strong_dls], strong_grp_ids)
weak_ipsi_df = df_for_stats(beh_avg_weak_dls_ipsi_day0.mice, day0_contra_dls_norms_conmod[weak_dls], weak_grp_ids)
ipsi_df_dls = [strong_ipsi_df; weak_ipsi_df]
ipsi_df_dls[!, :mc_neural_strength] = ipsi_df_dls.neural_strength .- mean(ipsi_df_dls.neural_strength);
frm = @formula(contrast_weight ~ day_split3*mc_neural_strength + (1 + day_split3|mouse))
contrasts = Dict(
    :mouse => DummyCoding(), 
    :day_split3 => DummyCoding()
)
ipsi_model = lme(frm, ipsi_df_dls, contrasts=contrasts)
anova(ipsi_model, type=3)

# figure 3.G
bias_dls_strong_day0 = hcat([behavior_weight[f].avg[:, 1] * (  beh_multiplier_contra[f]) for f in strong_grp_ids]...)
bias_dls_weak_day0 = hcat([behavior_weight[f].avg[:, 1] * ( beh_multiplier_contra[f]) for f in weak_grp_ids]...)
strong_bias_avg = nanmean(bias_dls_strong_day0, 2)
weak_bias_avg = nanmean(bias_dls_weak_day0, 2)
strong_bias_err = nansem(bias_dls_strong_day0, 2)
weak_bias_err = nansem(bias_dls_weak_day0, 2)

fig, ax = plt.subplots()
ax.plot(xt, strong_bias_avg, color=strong_color, label="Strong")
ax.fill_between(xt, strong_bias_avg .- strong_bias_err, 
                    strong_bias_avg .+ strong_bias_err, 
                    alpha=0.4, color=strong_color, label="_nolegend_")

ax.plot(xt, weak_bias_avg, color=weak_color, label="Weak")
ax.fill_between(xt, weak_bias_avg .- weak_bias_err, 
                    weak_bias_avg .+ weak_bias_err, 
                    alpha=0.4, color=weak_color, label="_nolegend_")

ax.set_xticks([1, 10, 20])
ax.set_xlabel("Session")
plt.savefig("strong_weak_choice_weights_bias_dls.pdf", bbox_inches="tight", transparent=true)
plt.close()

strong_bias_df = df_for_stats(bias_dls_strong_day0, day0_contra_dls_norms_conmod[strong_dls], strong_grp_ids) 
weak_bias_df = df_for_stats(bias_dls_weak_day0, day0_contra_dls_norms_conmod[weak_dls], weak_grp_ids)
bias_df_dls = [strong_bias_df; weak_bias_df]
bias_df_dls[!, :mc_neural_strength] = bias_df_dls.neural_strength .- mean(bias_df_dls.neural_strength);
frm = @formula(contrast_weight ~ day_split3*mc_neural_strength + (1 + day_split3|mouse))
contrasts = Dict(
    :mouse => DummyCoding(), 
    :day_split3 => DummyCoding()
)
bias_contra_model = lme(frm, bias_df_dls, contrasts=contrasts)
anova(bias_contra_model, type=3)


# figure 3.H
chist_dls_strong_day0 = hcat([behavior_weight[f].avg[:, 4] for f in strong_grp_ids]...)
chist_dls_weak_day0 = hcat([behavior_weight[f].avg[:, 4] for f in weak_grp_ids]...)
avg_chist_dls_strong_day0 = nanmean(chist_dls_strong_day0, 2)
avg_chist_dls_weak_day0 = nanmean(chist_dls_weak_day0, 2)
err_chist_dls_strong_day0 = nansem(chist_dls_strong_day0, 2)
err_chist_dls_weak_day0 = nansem(chist_dls_weak_day0, 2)

fig, ax = plt.subplots()
ax.plot(xt, avg_chist_dls_strong_day0, color=strong_color, label="Strong")
ax.fill_between(xt, avg_chist_dls_strong_day0 .- err_chist_dls_strong_day0,
                    avg_chist_dls_strong_day0 .+ err_chist_dls_strong_day0,
                    alpha=0.4, color=strong_color, label="_nolegend_")

ax.plot(xt, avg_chist_dls_weak_day0, color=weak_color, label="Weak")
ax.fill_between(xt, avg_chist_dls_weak_day0 .- err_chist_dls_weak_day0,
                    avg_chist_dls_weak_day0 .+ err_chist_dls_weak_day0,
                    alpha=0.4, color=weak_color, label="_nolegend_")

ax.set_xticks([1, 10, 20])
ax.set_xlabel("Session")
plt.savefig("strong_weak_choice_weights_chist_dls.pdf", bbox_inches="tight", transparent=true)
plt.close()

strong_chist_df = df_for_stats(chist_dls_strong_day0, day0_contra_dls_norms_conmod[strong_dls], strong_grp_ids)
weak_chist_df = df_for_stats(chist_dls_weak_day0, day0_contra_dls_norms_conmod[weak_dls], weak_grp_ids)

chist_df_dls = [strong_chist_df; weak_chist_df]
chist_df_dls[!, :mc_neural_strength] = chist_df_dls.neural_strength .- mean(chist_df_dls.neural_strength);
frm = @formula(contrast_weight ~ day_split3*mc_neural_strength + (1 + day_split3|mouse))
contrasts = Dict(
    :mouse => DummyCoding(), 
    :day_split3 => DummyCoding()
)
chist_contra_model = lme(frm, chist_df_dls, contrasts=contrasts)
anova(chist_contra_model, type=3)


"""
NAcc version
"""

day0_contra_nacc_norms = hcat([neu_day0_results.kernel_norm[f]["NAcc"][dms_ipsi_map[f]] for f in day0_mice]...)'
day0_contra_nacc_norms_conmod = day0_contra_nacc_norms[:, 4] .- day0_contra_nacc_norms[:, 1]
strong_nacc = day0_contra_nacc_norms_conmod .>= median(day0_contra_nacc_norms_conmod)
weak_nacc = .!strong_nacc
strong_color="darkblue"
weak_color="lightblue"
bins_strong = range(0, stop=maximum(day0_contra_nacc_norms_conmod), length=6)  # Bins for strong (>= 0)
bins_weak = range(minimum(day0_contra_nacc_norms_conmod), stop=median(day0_contra_nacc_norms_conmod), length=6)  # Bins for weak (< 0)

fig, ax = plt.subplots()
ax.hist(
    day0_contra_nacc_norms_conmod[strong_nacc], 
    bins=bins_strong, color=strong_color, edgecolor="white", alpha=1, label="Strong"
    )
ax.hist(
    day0_contra_nacc_norms_conmod[weak_nacc],
     bins=bins_weak, color=weak_color, edgecolor="white", alpha=1, label="Weak"
    )
ax.set_ylabel("Counts")
ax.axvline(median(day0_contra_nacc_norms_conmod), lw=2, linestyle="--", color="black", label="_nolegend_")
ax.set_xticks([-1, 0, 1])
plt.legend()
plt.savefig("strong_weak_nacc_day0_hist.pdf", transparent=true, bbox_inches="tight")




strong_grp_ids = collect(26:43)[strong_nacc]
weak_grp_ids = collect(26:43)[weak_nacc]

beh_avg_strong_nacc_contra_day0 = weight_by_group(behavior_weight, strong_grp_ids, "contra", "NAcc")
beh_avg_weak_nacc_contra_day0 = weight_by_group(behavior_weight, weak_grp_ids, "contra", "NAcc")

# figure 3.E
fig, ax = plt.subplots()
xt = 1:20
ax.plot(xt, beh_avg_strong_nacc_contra_day0.avg, color=strong_color, alpha=1)
ax.fill_between(xt, beh_avg_strong_nacc_contra_day0.avg .- beh_avg_strong_nacc_contra_day0.sem, 
                     beh_avg_strong_nacc_contra_day0.avg .+ beh_avg_strong_nacc_contra_day0.sem, 
                     alpha=0.5, color=strong_color, label="_nolegend_")

ax.plot(xt, beh_avg_weak_nacc_contra_day0.avg, color=weak_color)
ax.fill_between(xt, beh_avg_weak_nacc_contra_day0.avg .- beh_avg_weak_nacc_contra_day0.sem, 
                     beh_avg_weak_nacc_contra_day0.avg .+ beh_avg_weak_nacc_contra_day0.sem, 
                     alpha=0.5, color=weak_color, label="_nolegend_", )

ax.set_xticks([1, 10, 20])
ax.set_yticks([0, 4, 8])
ax.set_xlabel("Session")
plt.savefig("strong_weak_choice_weights_contra_nacc.pdf", bbox_inches="tight", transparent=true)
plt.close()

# check stats 
strong_contra_nacc_df = df_for_stats(beh_avg_strong_nacc_contra_day0.mice, day0_contra_nacc_norms_conmod[strong_nacc], strong_grp_ids)
weak_contra_nacc_df = df_for_stats(beh_avg_weak_nacc_contra_day0.mice, day0_contra_nacc_norms_conmod[weak_nacc], weak_grp_ids)
contra_df_nacc = [strong_contra_nacc_df; weak_contra_nacc_df]
contra_df_nacc[!, :mc_neural_strength] = contra_df_nacc.neural_strength .- mean(contra_df_nacc.neural_strength);
frm = @formula(contrast_weight ~ day_split3*mc_neural_strength + (1 + day_split3|mouse))
contrasts = Dict(
    :mouse => DummyCoding(), 
    :day_split3 => DummyCoding()
)
contra_model = lme(frm, contra_df_nacc, contrasts=contrasts)
anova(contra_model, type=3)

# figure 3.F
beh_avg_strong_nacc_ipsi_day0 = weight_by_group(behavior_weight, strong_grp_ids, "ipsi", "NAcc")
beh_avg_weak_nacc_ipsi_day0 = weight_by_group(behavior_weight, weak_grp_ids, "ipsi", "NAcc")

fig, ax = plt.subplots()
ax.plot(xt, beh_avg_strong_nacc_ipsi_day0.avg, color=strong_color, label="Strong")
ax.fill_between(xt, beh_avg_strong_nacc_ipsi_day0.avg .- beh_avg_strong_nacc_ipsi_day0.sem, 
                     beh_avg_strong_nacc_ipsi_day0.avg .+ beh_avg_strong_nacc_ipsi_day0.sem, 
                     alpha=0.4, color=strong_color, label="_nolegend_")

ax.plot(xt, beh_avg_weak_nacc_ipsi_day0.avg, color=weak_color, label="Weak")
ax.fill_between(xt, beh_avg_weak_nacc_ipsi_day0.avg .- beh_avg_weak_nacc_ipsi_day0.sem, 
                     beh_avg_weak_nacc_ipsi_day0.avg .+ beh_avg_weak_nacc_ipsi_day0.sem, 
                     alpha=0.4, color=weak_color, label="_nolegend_")

ax.set_xticks([1, 10, 20])
ax.set_yticks([0, 4, 8])
ax.set_xlabel("Session")
plt.savefig("strong_weak_choice_weights_ipsi_nacc.pdf", bbox_inches="tight", transparent=true)
plt.close()

# need to make the dataframes and do the stats like i do for all the other figure 3.X
strong_ipsi_df = df_for_stats(beh_avg_strong_nacc_ipsi_day0.mice, day0_contra_nacc_norms_conmod[strong_nacc], strong_grp_ids)
weak_ipsi_df = df_for_stats(beh_avg_weak_nacc_ipsi_day0.mice, day0_contra_nacc_norms_conmod[weak_nacc], weak_grp_ids)
ipsi_df_nacc = [strong_ipsi_df; weak_ipsi_df]
ipsi_df_nacc[!, :mc_neural_strength] = ipsi_df_nacc.neural_strength .- mean(ipsi_df_nacc.neural_strength);
frm = @formula(contrast_weight ~ day_split3*mc_neural_strength + (1 + day_split3|mouse))
contrasts = Dict(
    :mouse => DummyCoding(), 
    :day_split3 => DummyCoding()
)
ipsi_model = lme(frm, ipsi_df_nacc, contrasts=contrasts)
anova(ipsi_model, type=3)

# figure 3.G
bias_nacc_strong_day0 = hcat([behavior_weight[f].avg[:, 1] * (  beh_multiplier_contra[f]) for f in strong_grp_ids]...)
bias_nacc_weak_day0 = hcat([behavior_weight[f].avg[:, 1] * ( beh_multiplier_contra[f]) for f in weak_grp_ids]...)
strong_bias_avg = nanmean(bias_nacc_strong_day0, 2)
weak_bias_avg = nanmean(bias_nacc_weak_day0, 2)
strong_bias_err = nansem(bias_nacc_strong_day0, 2)
weak_bias_err = nansem(bias_nacc_weak_day0, 2)

fig, ax = plt.subplots()
ax.plot(xt, strong_bias_avg, color=strong_color, label="Strong")
ax.fill_between(xt, strong_bias_avg .- strong_bias_err, 
                    strong_bias_avg .+ strong_bias_err, 
                    alpha=0.4, color=strong_color, label="_nolegend_")

ax.plot(xt, weak_bias_avg, color=weak_color, label="Weak")
ax.fill_between(xt, weak_bias_avg .- weak_bias_err, 
                    weak_bias_avg .+ weak_bias_err, 
                    alpha=0.4, color=weak_color, label="_nolegend_")

ax.set_xticks([1, 10, 20])
ax.set_xlabel("Session")
plt.savefig("strong_weak_choice_weights_bias_nacc.pdf", bbox_inches="tight", transparent=true)
plt.close()

strong_bias_df = df_for_stats(bias_nacc_strong_day0, day0_contra_nacc_norms_conmod[strong_nacc], strong_grp_ids) 
weak_bias_df = df_for_stats(bias_nacc_weak_day0, day0_contra_nacc_norms_conmod[weak_nacc], weak_grp_ids)
bias_df_nacc = [strong_bias_df; weak_bias_df]
bias_df_nacc[!, :mc_neural_strength] = bias_df_nacc.neural_strength .- mean(bias_df_nacc.neural_strength);
frm = @formula(contrast_weight ~ day_split3*mc_neural_strength + (1 + day_split3|mouse))
contrasts = Dict(
    :mouse => DummyCoding(), 
    :day_split3 => DummyCoding()
)
bias_contra_model = lme(frm, bias_df_nacc, contrasts=contrasts)
anova(bias_contra_model, type=3)


# figure 3.H
chist_nacc_strong_day0 = hcat([behavior_weight[f].avg[:, 4] for f in strong_grp_ids]...)
chist_nacc_weak_day0 = hcat([behavior_weight[f].avg[:, 4] for f in weak_grp_ids]...)
avg_chist_nacc_strong_day0 = nanmean(chist_nacc_strong_day0, 2)
avg_chist_nacc_weak_day0 = nanmean(chist_nacc_weak_day0, 2)
err_chist_nacc_strong_day0 = nansem(chist_nacc_strong_day0, 2)
err_chist_nacc_weak_day0 = nansem(chist_nacc_weak_day0, 2)

fig, ax = plt.subplots()
ax.plot(xt, avg_chist_nacc_strong_day0, color=strong_color, label="Strong")
ax.fill_between(xt, avg_chist_nacc_strong_day0 .- err_chist_nacc_strong_day0,
                    avg_chist_nacc_strong_day0 .+ err_chist_nacc_strong_day0,
                    alpha=0.4, color=strong_color, label="_nolegend_")

ax.plot(xt, avg_chist_nacc_weak_day0, color=weak_color, label="Weak")
ax.fill_between(xt, avg_chist_nacc_weak_day0 .- err_chist_nacc_weak_day0,
                    avg_chist_nacc_weak_day0 .+ err_chist_nacc_weak_day0,
                    alpha=0.4, color=weak_color, label="_nolegend_")

ax.set_xticks([1, 10, 20])
ax.set_xlabel("Session")
plt.savefig("strong_weak_choice_weights_chist_nacc.pdf", bbox_inches="tight", transparent=true)
plt.close()

strong_chist_df = df_for_stats(chist_nacc_strong_day0, day0_contra_nacc_norms_conmod[strong_nacc], strong_grp_ids)
weak_chist_df = df_for_stats(chist_nacc_weak_day0, day0_contra_nacc_norms_conmod[weak_nacc], weak_grp_ids)

chist_df_nacc = [strong_chist_df; weak_chist_df]
chist_df_nacc[!, :mc_neural_strength] = chist_df_nacc.neural_strength .- mean(chist_df_nacc.neural_strength);
frm = @formula(contrast_weight ~ day_split3*mc_neural_strength + (1 + day_split3|mouse))
contrasts = Dict(
    :mouse => DummyCoding(), 
    :day_split3 => DummyCoding()
)
chist_contra_model = lme(frm, chist_df_nacc, contrasts=contrasts)
anova(chist_contra_model, type=3)
