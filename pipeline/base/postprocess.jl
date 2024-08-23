include(joinpath(@__DIR__, "preprocess.jl"))
include(joinpath(@__DIR__, "constants.jl"))
include(joinpath(@__DIR__, "correlation_functions.jl"))
include(joinpath(@__DIR__, "utility.jl"))

using JLD2;
using AnovaBase;
using AnovaMixedModels;

# load results
neu_results = load("/jukebox/witten/yoel/saved_results/neural_results.jld2", "results");
kernel_norm = neu_results.kernel_norm;
mouse_ids = [collect(13:16); collect(26:43)];

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
behavior_weight = load("/jukebox/witten/yoel/saved_results/choice_weights.jld2", "results")

# this replicates figure 2 d
n_days = 20
n_sim = 4
n_mice = 22
contra_stim_dms = [kernel_norm[mouse]["DMS"][dms_contra_map[mouse]] for mouse in mouse_ids];
contra_stim_dms = reshape(hcat(contra_stim_dms...), n_days, n_sim, n_mice);
avg_contra_dms = nanmean(contra_stim_dms, 3);

ipsi_stim_dms = [kernel_norm[mouse]["DMS"][dms_ipsi_map[mouse]] for mouse in mouse_ids];
ipsi_stim_dms = reshape(hcat(ipsi_stim_dms...), n_days, n_sim, n_mice);
avg_ipsi_dms = nanmean(ipsi_stim_dms, 3);

# need to do the same as above for DLS and NAcc
contra_stim_dls = [kernel_norm[mouse]["DLS"][dms_ipsi_map[mouse]] for mouse in mouse_ids];
contra_stim_dls = reshape(hcat(contra_stim_dls...), n_days, n_sim, n_mice);
avg_contra_dls = nanmean(contra_stim_dls, 3);

ipsi_stim_dls = [kernel_norm[mouse]["DLS"][dms_contra_map[mouse]] for mouse in mouse_ids];
ipsi_stim_dls = reshape(hcat(ipsi_stim_dls...), n_days, n_sim, n_mice);
avg_ipsi_dls = nanmean(ipsi_stim_dls, 3);

contra_stim_nacc = [kernel_norm[mouse]["NAcc"][dms_ipsi_map[mouse]] for mouse in mouse_ids];
contra_stim_nacc = reshape(hcat(contra_stim_nacc...), n_days, n_sim, n_mice);
avg_contra_nacc = nanmean(contra_stim_nacc, 3);

ipsi_stim_nacc = [kernel_norm[mouse]["NAcc"][dms_contra_map[mouse]] for mouse in mouse_ids];
ipsi_stim_nacc = reshape(hcat(ipsi_stim_nacc...), n_days, n_sim, n_mice);
avg_ipsi_nacc = nanmean(ipsi_stim_nacc, 3);


# these replicate figures 2 e
contra_dms_corrs = avg_contra_correlation_conmod_dms(kernel_norm, behavior_weight, mouse_ids)
contra_dls_corrs = avg_contra_correlation_conmod_dls(kernel_norm, behavior_weight, mouse_ids)
contra_nacc_corrs = avg_contra_correlation_conmod_nacc(kernel_norm, behavior_weight, mouse_ids)

ipsi_dms_corrs = avg_ipsi_correlation_conmod_dms(kernel_norm, behavior_weight, mouse_ids)
ipsi_dls_corrs = avg_ipsi_correlation_conmod_dls(kernel_norm, behavior_weight, mouse_ids)
ipsi_nacc_corrs = avg_ipsi_correlation_conmod_nacc(kernel_norm, behavior_weight, mouse_ids)

# load day0 results
neu_day0_results = load("/jukebox/witten/yoel/saved_results/day0_neural_results.jld2", "results");

day0_mice = 26:43;
# the averages  of these kernels replicate figure 3.B
contra_dms_kernel = [neu_day0_results.kernel[f]["DMS"][dms_contra_map[f]] for f in day0_mice];
ipsi_dms_kernel = [neu_day0_results.kernel[f]["DMS"][dms_ipsi_map[f]] for f in day0_mice]

contra_dls_kernel = [neu_day0_results.kernel[f]["DLS"][dms_ipsi_map[f]] for f in day0_mice];
ipsi_dls_kernel = [neu_day0_results.kernel[f]["DLS"][dms_contra_map[f]] for f in day0_mice];

contra_nacc_kernel = [neu_day0_results.kernel[f]["NAcc"][dms_ipsi_map[f]] for f in day0_mice];
ipsi_nacc_kernel = [neu_day0_results.kernel[f]["NAcc"][dms_contra_map[f]] for f in day0_mice];

# figure 3.D
day0_contra_dms_norms = hcat([neu_day0_results.kernel_norm[f]["DMS"][dms_contra_map[f]] for f in day0_mice]...)'
day0_contra_dms_norms_conmod = day0_contra_dms_norms[:, 4] .- day0_contra_dms_norms[:, 1]
plt.hist(day0_contra_dms_norms_conmod, edgecolor="white")

strong_dms = day0_contra_dms_norms_conmod .> 2
weak_dms = .!strong_dms

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

    group_weights = hcat([
        behavior_weight[f].avg[:, side_map[f]] * (multiplier * beh_multiplier_contra[f]) for f in group_ids
        ]...)

    return (
        mice = group_weights,
        avg = nanmean(group_weights, 2),
        sem = nansem(group_weights, 2)
    )

end



strong_grp_ids = collect(26:43)[strong_dms]
weak_grp_ids = collect(26:43)[weak_dms]

beh_avg_strong_dms_contra_day0 = weight_by_group(behavior_weight, strong_grp_ids, "contra", "DMS")
beh_avg_weak_dms_contra_day0 = weight_by_group(behavior_weight, weak_grp_ids, "contra", "DMS")


# figure 3.E
xt = 1:20
plt.plot(xt, beh_avg_strong_dms_contra_day0.avg)
plt.fill_between(xt, beh_avg_strong_dms_contra_day0.avg .- beh_avg_strong_dms_contra_day0.sem, 
                     beh_avg_strong_dms_contra_day0.avg .+ beh_avg_strong_dms_contra_day0.sem, alpha=0.4)
plt.plot(xt, beh_avg_weak_dms_contra_day0.avg)
plt.fill_between(xt, beh_avg_weak_dms_contra_day0.avg .- beh_avg_weak_dms_contra_day0.sem, 
                     beh_avg_weak_dms_contra_day0.avg .+ beh_avg_weak_dms_contra_day0.sem, alpha=0.4)


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

xt = 1:20
plt.plot(xt, beh_avg_strong_dms_ipsi_day0.avg)
plt.fill_between(xt, beh_avg_strong_dms_ipsi_day0.avg .- beh_avg_strong_dms_ipsi_day0.sem, 
                     beh_avg_strong_dms_ipsi_day0.avg .+ beh_avg_strong_dms_ipsi_day0.sem, alpha=0.4)
plt.plot(xt, beh_avg_weak_dms_ipsi_day0.avg)
plt.fill_between(xt, beh_avg_weak_dms_ipsi_day0.avg .- beh_avg_weak_dms_ipsi_day0.sem, 
                     beh_avg_weak_dms_ipsi_day0.avg .+ beh_avg_weak_dms_ipsi_day0.sem, alpha=0.4)


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
plt.plot(xt, strong_bias_avg)
plt.fill_between(xt, strong_bias_avg .- strong_bias_err, strong_bias_avg .+ strong_bias_err, alpha=0.4)
plt.plot(xt, weak_bias_avg)
plt.fill_between(xt, weak_bias_avg .- weak_bias_err, weak_bias_avg .+ weak_bias_err, alpha=0.4)

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

plt.plot(nanmean(chist_dms_strong_day0, 2))
plt.plot(nanmean(chist_dms_weak_day0, 2))

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