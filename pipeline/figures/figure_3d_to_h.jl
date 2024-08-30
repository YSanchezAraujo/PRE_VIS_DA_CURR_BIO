include(joinpath(@__DIR__, "../base/preprocess.jl"));
include(joinpath(@__DIR__, "../base/constants.jl"));
include(joinpath(@__DIR__, "../base/correlation_functions.jl"));
include(joinpath(@__DIR__, "../base/utility.jl"));
include(joinpath(@__DIR__, "../base/figboilerplate.jl"));
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
day0_contra_dms_norms = hcat([neu_day0_results.kernel_norm[f]["DMS"][dms_contra_map[f]] for f in day0_mice]...)'
day0_contra_dms_norms_conmod = day0_contra_dms_norms[:, 4] .- day0_contra_dms_norms[:, 1]
strong_dms = day0_contra_dms_norms_conmod .> median(day0_contra_dms_norms_conmod)
weak_dms = .!strong_dms
strong_color="tab:red"
weak_color="tab:orange"
#bins = range(minimum(day0_contra_dms_norms_conmod), stop=maximum(day0_contra_dms_norms_conmod), length=13)
bin_width = 0.7  # Example bin width; adjust as necessary
bins_strong = collect(median(day0_contra_dms_norms_conmod):bin_width:maximum(day0_contra_dms_norms_conmod) + bin_width)
bins_weak = collect(minimum(day0_contra_dms_norms_conmod):bin_width:median(day0_contra_dms_norms_conmod) + bin_width)

fig, ax = plt.subplots()
ax.hist(
    day0_contra_dms_norms_conmod[strong_dms], 
    bins=bins_strong, color=strong_color, edgecolor="white", alpha=1, label="Strong", 
    )
ax.hist(
    day0_contra_dms_norms_conmod[weak_dms],
     bins=bins_weak, color=weak_color, edgecolor="white", alpha=1, label="Weak",
    )
ax.set_ylabel("Counts")
ax.axvline(median(day0_contra_dms_norms_conmod), lw=2, linestyle="--", color="black", label="_nolegend_")
ax.set_xticks([0, 2, 4, 6])
plt.legend()
plt.savefig("strong_weak_dms_day0_hist.pdf", transparent=true, bbox_inches="tight")

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

strong_grp_ids = collect(26:43)[strong_dms]
weak_grp_ids = collect(26:43)[weak_dms]

beh_avg_strong_dms_contra_day0 = weight_by_group(behavior_weight, strong_grp_ids, "contra", "DMS")
beh_avg_weak_dms_contra_day0 = weight_by_group(behavior_weight, weak_grp_ids, "contra", "DMS")

# figure 3.E
fig, ax = plt.subplots()
xt = 1:20
ax.plot(xt, beh_avg_strong_dms_contra_day0.avg, color=strong_color, alpha=1)
ax.fill_between(xt, beh_avg_strong_dms_contra_day0.avg .- beh_avg_strong_dms_contra_day0.sem, 
                     beh_avg_strong_dms_contra_day0.avg .+ beh_avg_strong_dms_contra_day0.sem, 
                     alpha=0.5, color=strong_color, label="_nolegend_")

ax.plot(xt, beh_avg_weak_dms_contra_day0.avg, color=weak_color)
ax.fill_between(xt, beh_avg_weak_dms_contra_day0.avg .- beh_avg_weak_dms_contra_day0.sem, 
                     beh_avg_weak_dms_contra_day0.avg .+ beh_avg_weak_dms_contra_day0.sem, 
                     alpha=0.5, color=weak_color, label="_nolegend_", )

ax.set_xticks([1, 10, 20])
ax.set_yticks([0, 4, 8])
ax.set_xlabel("Session")
plt.savefig("strong_weak_choice_weights_contra_dms.pdf", bbox_inches="tight", transparent=true)
plt.close()

# check stats 
strong_contra_dms_df = df_for_stats(beh_avg_strong_dms_contra_day0.mice, day0_contra_dms_norms_conmod[strong_dms], strong_grp_ids)
weak_contra_dms_df = df_for_stats(beh_avg_weak_dms_contra_day0.mice, day0_contra_dms_norms_conmod[weak_dms], weak_grp_ids)
contra_df_dms = [strong_contra_dms_df; weak_contra_dms_df]
contra_df_dms[!, :mc_neural_strength] = contra_df_dms.neural_strength .- mean(contra_df_dms.neural_strength);
frm = @formula(contrast_weight ~ day_split3*mc_neural_strength + (1 + day_split3|mouse))
contrasts = Dict(
    :mouse => DummyCoding(), 
    :day_split3 => DummyCoding()
)
contra_model = lme(frm, contra_df_dms, contrasts=contrasts)
anova(contra_model, type=3)

# figure 3.F
beh_avg_strong_dms_ipsi_day0 = weight_by_group(behavior_weight, strong_grp_ids, "ipsi", "DMS")
beh_avg_weak_dms_ipsi_day0 = weight_by_group(behavior_weight, weak_grp_ids, "ipsi", "DMS")

fig, ax = plt.subplots()
ax.plot(xt, beh_avg_strong_dms_ipsi_day0.avg, color=strong_color, label="Strong")
ax.fill_between(xt, beh_avg_strong_dms_ipsi_day0.avg .- beh_avg_strong_dms_ipsi_day0.sem, 
                     beh_avg_strong_dms_ipsi_day0.avg .+ beh_avg_strong_dms_ipsi_day0.sem, 
                     alpha=0.4, color=strong_color, label="_nolegend_")

ax.plot(xt, beh_avg_weak_dms_ipsi_day0.avg, color=weak_color, label="Weak")
ax.fill_between(xt, beh_avg_weak_dms_ipsi_day0.avg .- beh_avg_weak_dms_ipsi_day0.sem, 
                     beh_avg_weak_dms_ipsi_day0.avg .+ beh_avg_weak_dms_ipsi_day0.sem, 
                     alpha=0.4, color=weak_color, label="_nolegend_")

ax.set_xticks([1, 10, 20])
ax.set_yticks([0, 4, 8])
ax.set_xlabel("Session")
plt.savefig("strong_weak_choice_weights_ipsi_dms.pdf", bbox_inches="tight", transparent=true)
plt.close()

# need to make the dataframes and do the stats like i do for all the other figure 3.X
strong_ipsi_df = df_for_stats(beh_avg_strong_dms_ipsi_day0.mice, day0_contra_dms_norms_conmod[strong_dms], strong_grp_ids)
weak_ipsi_df = df_for_stats(beh_avg_weak_dms_ipsi_day0.mice, day0_contra_dms_norms_conmod[weak_dms], weak_grp_ids)
ipsi_df_dms = [strong_ipsi_df; weak_ipsi_df]
ipsi_df_dms[!, :mc_neural_strength] = ipsi_df_dms.neural_strength .- mean(ipsi_df_dms.neural_strength);
frm = @formula(contrast_weight ~ day_split3*mc_neural_strength + (1 + day_split3|mouse))
contrasts = Dict(
    :mouse => DummyCoding(), 
    :day_split3 => DummyCoding()
)
ipsi_model = lme(frm, ipsi_df_dms, contrasts=contrasts)
anova(ipsi_model, type=3)

# figure 3.G
bias_dms_strong_day0 = hcat([behavior_weight[f].avg[:, 1] * (  beh_multiplier_contra[f]) for f in strong_grp_ids]...)
bias_dms_weak_day0 = hcat([behavior_weight[f].avg[:, 1] * ( beh_multiplier_contra[f]) for f in weak_grp_ids]...)
strong_bias_avg = nanmean(bias_dms_strong_day0, 2)
weak_bias_avg = nanmean(bias_dms_weak_day0, 2)
strong_bias_err = nansem(bias_dms_strong_day0, 2)
weak_bias_err = nansem(bias_dms_weak_day0, 2)

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
plt.savefig("strong_weak_choice_weights_bias_dms.pdf", bbox_inches="tight", transparent=true)
plt.close()

strong_bias_df = df_for_stats(bias_dms_strong_day0, day0_contra_dms_norms_conmod[strong_dms], strong_grp_ids) 
weak_bias_df = df_for_stats(bias_dms_weak_day0, day0_contra_dms_norms_conmod[weak_dms], weak_grp_ids)
bias_df_dms = [strong_bias_df; weak_bias_df]
bias_df_dms[!, :mc_neural_strength] = bias_df_dms.neural_strength .- mean(bias_df_dms.neural_strength);
frm = @formula(contrast_weight ~ day_split3*mc_neural_strength + (1 + day_split3|mouse))
contrasts = Dict(
    :mouse => DummyCoding(), 
    :day_split3 => DummyCoding()
)
bias_contra_model = lme(frm, bias_df_dms, contrasts=contrasts)
anova(bias_contra_model, type=3)


# figure 3.H
chist_dms_strong_day0 = hcat([behavior_weight[f].avg[:, 4] for f in strong_grp_ids]...)
chist_dms_weak_day0 = hcat([behavior_weight[f].avg[:, 4] for f in weak_grp_ids]...)
avg_chist_dms_strong_day0 = nanmean(chist_dms_strong_day0, 2)
avg_chist_dms_weak_day0 = nanmean(chist_dms_weak_day0, 2)
err_chist_dms_strong_day0 = nansem(chist_dms_strong_day0, 2)
err_chist_dms_weak_day0 = nansem(chist_dms_weak_day0, 2)

fig, ax = plt.subplots()
ax.plot(xt, avg_chist_dms_strong_day0, color=strong_color, label="Strong")
ax.fill_between(xt, avg_chist_dms_strong_day0 .- err_chist_dms_strong_day0,
                    avg_chist_dms_strong_day0 .+ err_chist_dms_strong_day0,
                    alpha=0.4, color=strong_color, label="_nolegend_")

ax.plot(xt, avg_chist_dms_weak_day0, color=weak_color, label="Weak")
ax.fill_between(xt, avg_chist_dms_weak_day0 .- err_chist_dms_weak_day0,
                    avg_chist_dms_weak_day0 .+ err_chist_dms_weak_day0,
                    alpha=0.4, color=weak_color, label="_nolegend_")

ax.set_xticks([1, 10, 20])
ax.set_xlabel("Session")
plt.savefig("strong_weak_choice_weights_chist_dms.pdf", bbox_inches="tight", transparent=true)
plt.close()

strong_chist_df = df_for_stats(chist_dms_strong_day0, day0_contra_dms_norms_conmod[strong_dms], strong_grp_ids)
weak_chist_df = df_for_stats(chist_dms_weak_day0, day0_contra_dms_norms_conmod[weak_dms], weak_grp_ids)

chist_df_dms = [strong_chist_df; weak_chist_df]
chist_df_dms[!, :mc_neural_strength] = chist_df_dms.neural_strength .- mean(chist_df_dms.neural_strength);
frm = @formula(contrast_weight ~ day_split3*mc_neural_strength + (1 + day_split3|mouse))
contrasts = Dict(
    :mouse => DummyCoding(), 
    :day_split3 => DummyCoding()
)
chist_contra_model = lme(frm, chist_df_dms, contrasts=contrasts)
anova(chist_contra_model, type=3)
