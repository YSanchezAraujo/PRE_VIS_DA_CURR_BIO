include(joinpath(@__DIR__, "../base/preprocess.jl"));
include(joinpath(@__DIR__, "../base/constants.jl"));
include(joinpath(@__DIR__, "../base/design_matrix.jl"));
include(joinpath(@__DIR__, "../base/correlation_functions.jl"));
include(joinpath(@__DIR__, "../base/utility.jl"));
include(joinpath(@__DIR__, "../base/figboilerplate.jl"));
include(joinpath(@__DIR__, "../base/encoding_model.jl"));


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

function make_nan_from_qc_d0(kernel_norm, qc_dict, mouse_ids)
    for mouse in mouse_ids
        for evn in event_names[1:2]
            for reg in ["NAcc", "DMS", "DLS"]
                qc_reg = qc_dict[mouse].pretrain[!, Symbol("QC_$(reg)")][1]
                if qc_reg == 0
                    kernel_norm[mouse][reg][evn] .= NaN
                end
            end
        end
    end

    return kernel_norm
end


function get_side_map_multiplier(side_label_beh, side_label_neu, region_label)
    # For DMS
    if side_label_beh == "contra" && side_label_neu == "contra" && region_label == "DMS"
        return beh_dms_contra_map, 1, dms_contra_map
    end

    if side_label_beh == "contra" && side_label_neu == "ipsi" && region_label == "DMS"
        return beh_dms_contra_map, 1, dms_ipsi_map
    end

    if side_label_beh == "ipsi" && side_label_neu == "contra" && region_label == "DMS"
        return beh_dms_ipsi_map, -1, dms_contra_map
    end

    if side_label_beh == "ipsi" && side_label_neu == "ipsi" && region_label == "DMS"
        return beh_dms_ipsi_map, -1, dms_ipsi_map
    end

    # For NAcc
    if side_label_beh == "contra" && side_label_neu == "contra" && region_label == "NAcc"
        return beh_dms_ipsi_map, -1, dms_ipsi_map
    end

    if side_label_beh == "contra" && side_label_neu == "ipsi" && region_label == "NAcc"
        return beh_dms_ipsi_map, -1, dms_contra_map
    end

    if side_label_beh == "ipsi" && side_label_neu == "contra" && region_label == "NAcc"
        return beh_dms_contra_map, 1, dms_ipsi_map
    end

    if side_label_beh == "ipsi" && side_label_neu == "ipsi" && region_label == "NAcc"
        return beh_dms_contra_map, 1, dms_contra_map
    end

    # For DLS (same structure as NAcc)
    if side_label_beh == "contra" && side_label_neu == "contra" && region_label == "DLS"
        return beh_dms_ipsi_map, -1, dms_ipsi_map
    end

    if side_label_beh == "contra" && side_label_neu == "ipsi" && region_label == "DLS"
        return beh_dms_ipsi_map, -1, dms_contra_map
    end

    if side_label_beh == "ipsi" && side_label_neu == "contra" && region_label == "DLS"
        return beh_dms_contra_map, 1, dms_ipsi_map
    end

    if side_label_beh == "ipsi" && side_label_neu == "ipsi" && region_label == "DLS"
        return beh_dms_contra_map, 1, dms_contra_map
    end
end


kernel_norm = make_nan_from_qc(kernel_norm, qc_dict, mouse_ids)
neu_day0_results = load("/jukebox/witten/yoel/saved_results/day0_neural_results.jld2", "results");
day0_mice = 26:43;
day0_kernel_norm = make_nan_from_qc_d0(neu_day0_results.kernel_norm, qc_dict, 26:43)


day0_kernel_norms_dms = hcat([day0_kernel_norm[f]["DMS"][dms_contra_map[f]] for f in day0_mice]...)'
day0_kernel_norms_conmod_dms = day0_kernel_norms_dms[:, 4] .- day0_kernel_norms_dms[:, 1];


function get_day0_side_with_beh_side(
    neu_day0_kernel_norm, 
    neu_kernel_norm, 
    behavior_weight, 
    mouse_ids, 
    region_label, 
    side_label_neu,
    side_label_beh
    )

    beh_side_map, multiplier, neu_side_map = get_side_map_multiplier(side_label_neu, side_label_beh, region_label)

    day0_kernel_norms = hcat([neu_day0_kernel_norm[f][region_label][neu_side_map[f]] for f in mouse_ids]...)'
    day0_kernel_norms_conmod = day0_kernel_norms[:, 4] .- day0_kernel_norms[:, 1];

    day120_kernel_norms = hcat([
        neu_kernel_norm[f][region_label][neu_side_map[f]][:, 4] .- 
        neu_kernel_norm[f][region_label][neu_side_map[f]][:, 1]
        for f in mouse_ids
        ]...)

    beh = hcat([
        behavior_weight[f].avg[:, beh_side_map[f]] * (multiplier * beh_multiplier_contra[f]) for f in mouse_ids
        ]...)

    return beh, day0_kernel_norms_conmod, day120_kernel_norms

end

# contra to contra correlations
dms_contra_beh, dms_contra_neu_d0, dms_contra_neu = get_day0_side_with_beh_side(
    day0_kernel_norm, kernel_norm, behavior_weight, day0_mice, "DMS", "contra", "contra"
    )

dls_contra_beh, dls_contra_neu_d0, dls_contra_neu = get_day0_side_with_beh_side(
    day0_kernel_norm, kernel_norm, behavior_weight, 5, day0_mice, "DLS", "contra", "contra"
    )

nacc_contra_beh, nacc_contra_neu_d0, nacc_contra_neu = get_day0_side_with_beh_side(
    day0_kernel_norm, kernel_norm, behavior_weight, 5, day0_mice, "NAcc", "contra", "contra"
    )

# ipsi to ipsi correlations
dms_ipsi_beh, dms_ipsi_neu_d0, dms_ipsi_neu = get_day0_side_with_beh_side(
    neu_day0_results, kernel_norm, behavior_weight, 5, day0_mice, "DMS", "ipsi", "ipsi"
    )

nacc_ipsi_beh, nacc_ipsi_neu_d0, nacc_ipsi_neu = get_day0_side_with_beh_side(
    neu_day0_results, kernel_norm, behavior_weight, 5, day0_mice, "NAcc", "ipsi", "ipsi"
    )

dls_ipsi_beh, dls_ipsi_neu_d0, dls_ipsi_neu = get_day0_side_with_beh_side(
    neu_day0_results, kernel_norm, behavior_weight, 5, day0_mice, "DLS", "ipsi", "ipsi"
    )


dms_ipsi_beh, dms_contra_neu_d0, dms_contra_neu = get_day0_side_with_beh_side(
    day0_kernel_norm, kernel_norm, behavior_weight, 5, day0_mice, "DMS", "contra", "ipsi"
    )

dls_ipsi_beh, dls_contra_neu_d0, dls_contra_neu = get_day0_side_with_beh_side(
    day0_kernel_norm, kernel_norm, behavior_weight, 5, day0_mice, "DLS", "contra", "ipsi"
    )

nacc_ipsi_beh, nacc_contra_neu_d0, nacc_contra_neu = get_day0_side_with_beh_side(
    day0_kernel_norm, kernel_norm, behavior_weight, 5, day0_mice, "NAcc", "contra", "ipsi"
    )    



function get_correlations_and_pvals(
    neu_day0_kernel_norm, 
    neu_kernel_norm, 
    behavior_weight, 
    mouse_ids, 
    region_label, 
    side_label_neu,
    side_label_beh
)

    side_beh, side_neu_d0, side_neu = get_day0_side_with_beh_side(
        neu_day0_kernel_norm, kernel_norm, behavior_weight, mouse_ids, region_label, side_label_neu, side_label_beh
        )

    cortraj = [   
        [robust_nanlm(side_neu_d0, nanmean(side_beh[16:20, :], 1)).cor];
        [robust_nanlm(side_neu[d, :], nanmean(side_beh[16:20, :], 1)).cor for d in 1:20]
    ];

    pvaltraj = [   
        [robust_nanlm(side_neu_d0, nanmean(side_beh[16:20, :], 1)).pval];
        [robust_nanlm(side_neu[d, :], nanmean(side_beh[16:20, :], 1)).pval for d in 1:20]
    ];

    return (
        beh = side_beh,
        neu = side_neu,
        d0neu = side_neu_d0,
        cortraj = cortraj,
        pvaltraj = pvaltraj
    )
end

# CONTRA CONTRA
dms_contra_contra = get_correlations_and_pvals(
    day0_kernel_norm, kernel_norm, behavior_weight, day0_mice, "DMS", "contra", "contra"
    )

dls_contra_contra = get_correlations_and_pvals(
    day0_kernel_norm, kernel_norm, behavior_weight, day0_mice, "DLS", "contra", "contra"
    )

nacc_contra_contra = get_correlations_and_pvals(
    day0_kernel_norm, kernel_norm, behavior_weight, day0_mice, "NAcc", "contra", "contra"
    )

# IPSI IPSI
dms_ipsi_ipsi = get_correlations_and_pvals(
    day0_kernel_norm, kernel_norm, behavior_weight, day0_mice, "DMS", "ipsi", "ipsi"
    )

dls_ipsi_ipsi = get_correlations_and_pvals(
    day0_kernel_norm, kernel_norm, behavior_weight, day0_mice, "DLS", "ipsi", "ipsi"
    )

nacc_ipsi_ipsi = get_correlations_and_pvals(
    day0_kernel_norm, kernel_norm, behavior_weight, day0_mice, "NAcc", "ipsi", "ipsi"
    )

# CONTRA IPSI
dms_contra_ipsi = get_correlations_and_pvals(
    day0_kernel_norm, kernel_norm, behavior_weight, day0_mice, "DMS", "contra", "ipsi"
    )

dls_contra_ipsi = get_correlations_and_pvals(
    day0_kernel_norm, kernel_norm, behavior_weight, day0_mice, "DLS", "contra", "ipsi"
    )

nacc_contra_ipsi = get_correlations_and_pvals(
    day0_kernel_norm, kernel_norm, behavior_weight, day0_mice, "NAcc", "contra", "ipsi"
    )

# IPSI CONTRA
dms_ipsi_contra = get_correlations_and_pvals(
    day0_kernel_norm, kernel_norm, behavior_weight, day0_mice, "DMS", "ipsi", "contra"
    )

dls_ipsi_contra = get_correlations_and_pvals(
    day0_kernel_norm, kernel_norm, behavior_weight, day0_mice, "DLS", "ipsi", "contra"
    )

nacc_ipsi_contra = get_correlations_and_pvals(
    day0_kernel_norm, kernel_norm, behavior_weight, day0_mice, "NAcc", "ipsi", "contra"
    )





fig, ax = plt.subplots(nrows=1, ncols=2, width_ratios=[0.1, 1.9], figsize=(5,4))
xt = 1:15
ax[1].plot([0], [nacc_contra_contra.cortraj[1]], "-o", lw=3)
ax[1].plot([0], [dms_contra_contra.cortraj[1]], "-o", lw=3)
ax[1].plot([0], [dls_contra_contra.cortraj[1]], "-o", lw=3)
ax[2].plot(xt, nacc_contra_contra.cortraj[2:16], "-o", lw=3)
ax[2].plot(xt, dms_contra_contra.cortraj[2:16], "-o", lw=3)
ax[2].plot(xt, dls_contra_contra.cortraj[2:16], "-o", lw=3)
ax[2].spines["left"].set_visible(false)
ax[2].set_yticks([])
ax[1].set_ylim(-0.7, 1.)
ax[2].set_ylim(-0.7, 1.)
ax[1].set_xlim(-0.01, 0.01)
ax[1].set_xticks([0])
ax[1].set_yticks([-0.5, 0, 0.5, 1])
plt.subplots_adjust(wspace=0.07)

ax[1].annotate(
    xy = [0, 0.8], text = pval_stars(nacc_contra_contra.pvaltraj[1]), rotation=90, color="tab:blue"
)

ax[1].annotate(
    xy = [0, 0.9], text = pval_stars(dms_contra_contra.pvaltraj[1]), rotation=90, color="tab:orange"
)

ax[1].annotate(
    xy = [0, 1], text = pval_stars(dls_contra_contra.pvaltraj[1]), rotation=90, color="tab:green"
)

for (day,j) in enumerate(2:16)
    ax[2].annotate(
        xy = [day-0.3, 0.8], text = pval_stars(nacc_contra_contra.pvaltraj[j]), rotation=90, color="tab:blue"
    )

    ax[2].annotate(
        xy = [day-0.3, 0.9], text = pval_stars(dms_contra_contra.pvaltraj[j]), rotation=90, color="tab:orange"
    )

    ax[2].annotate(
        xy = [day-0.3, 1], text = pval_stars(dls_contra_contra.pvaltraj[j]), rotation=90, color="tab:green"
    )
end

ax[2].set_xticks([1, 5, 10, 15])
ax[1].set_xticks([0])
ax[2].axhline(0, linestyle="--", color="black")
ax[1].axhline(0, linestyle="--", color="black")
plt.savefig("suppfig_contra_contra_traj.pdf", transparent=true, bbox_inches="tight")
    



fig, ax = plt.subplots(nrows=1, ncols=2, width_ratios=[0.1, 1.9], figsize=(5,4))
xt = 1:15
ax[1].plot([0], [nacc_ipsi_ipsi.cortraj[1]], "-o", lw=3)
ax[1].plot([0], [dms_ipsi_ipsi.cortraj[1]], "-o", lw=3)
ax[1].plot([0], [dls_ipsi_ipsi.cortraj[1]], "-o", lw=3)
ax[2].plot(xt, nacc_ipsi_ipsi.cortraj[2:16], "-o", lw=3)
ax[2].plot(xt, dms_ipsi_ipsi.cortraj[2:16], "-o", lw=3)
ax[2].plot(xt, dls_ipsi_ipsi.cortraj[2:16], "-o", lw=3)
ax[2].spines["left"].set_visible(false)
ax[2].set_yticks([])
ax[1].set_ylim(-0.7, 1.)
ax[2].set_ylim(-0.7, 1.)
ax[1].set_xlim(-0.01, 0.01)
ax[1].set_xticks([0])
ax[1].set_yticks([-0.5, 0, 0.5, 1])
plt.subplots_adjust(wspace=0.07)

ax[1].annotate(
    xy = [0, 0.8], text = pval_stars(nacc_ipsi_ipsi.pvaltraj[1]), rotation=90, color="tab:blue"
)

ax[1].annotate(
    xy = [0, 0.9], text = pval_stars(dms_ipsi_ipsi.pvaltraj[1]), rotation=90, color="tab:orange"
)

ax[1].annotate(
    xy = [0, 1], text = pval_stars(dls_ipsi_ipsi.pvaltraj[1]), rotation=90, color="tab:green"
)

for (day,j) in enumerate(2:16)
    ax[2].annotate(
        xy = [day-0.3, 0.8], text = pval_stars(nacc_ipsi_ipsi.pvaltraj[j]), rotation=90, color="tab:blue"
    )

    ax[2].annotate(
        xy = [day-0.3, 0.9], text = pval_stars(dms_ipsi_ipsi.pvaltraj[j]), rotation=90, color="tab:orange"
    )

    ax[2].annotate(
        xy = [day-0.3, 1], text = pval_stars(dls_ipsi_ipsi.pvaltraj[j]), rotation=90, color="tab:green"
    )
end

ax[2].set_xticks([1, 5, 10, 15])
ax[1].set_xticks([0])
ax[2].axhline(0, linestyle="--", color="black")
ax[1].axhline(0, linestyle="--", color="black")
plt.savefig("suppfig_ipsi_ipsi_traj.pdf", transparent=true, bbox_inches="tight")



fig, ax = plt.subplots(nrows=1, ncols=2, width_ratios=[0.1, 1.9], figsize=(5,4))
xt = 1:15
ax[1].plot([0], [nacc_contra_ipsi.cortraj[1]], "-o", lw=3)
ax[1].plot([0], [dms_contra_ipsi.cortraj[1]], "-o", lw=3)
ax[1].plot([0], [dls_contra_ipsi.cortraj[1]], "-o", lw=3)
ax[2].plot(xt, nacc_contra_ipsi.cortraj[2:16], "-o", lw=3)
ax[2].plot(xt, dms_contra_ipsi.cortraj[2:16], "-o", lw=3)
ax[2].plot(xt, dls_contra_ipsi.cortraj[2:16], "-o", lw=3)
ax[2].spines["left"].set_visible(false)
ax[2].set_yticks([])
ax[1].set_ylim(-0.7, 1.)
ax[2].set_ylim(-0.7, 1.)
ax[1].set_xlim(-0.01, 0.01)
ax[1].set_xticks([0])
ax[1].set_yticks([-0.5, 0, 0.5, 1])
plt.subplots_adjust(wspace=0.07)

ax[1].annotate(
    xy = [0, 0.8], text = pval_stars(nacc_contra_ipsi.pvaltraj[1]), rotation=90, color="tab:blue"
)

ax[1].annotate(
    xy = [0, 0.9], text = pval_stars(dms_contra_ipsi.pvaltraj[1]), rotation=90, color="tab:orange"
)

ax[1].annotate(
    xy = [0, 1], text = pval_stars(dls_contra_ipsi.pvaltraj[1]), rotation=90, color="tab:green"
)

for (day,j) in enumerate(2:16)
    ax[2].annotate(
        xy = [day-0.3, 0.8], text = pval_stars(nacc_contra_ipsi.pvaltraj[j]), rotation=90, color="tab:blue"
    )

    ax[2].annotate(
        xy = [day-0.3, 0.9], text = pval_stars(dms_contra_ipsi.pvaltraj[j]), rotation=90, color="tab:orange"
    )

    ax[2].annotate(
        xy = [day-0.3, 1], text = pval_stars(dls_contra_ipsi.pvaltraj[j]), rotation=90, color="tab:green"
    )
end

ax[2].set_xticks([1, 5, 10, 15])
ax[1].set_xticks([0])
ax[2].axhline(0, linestyle="--", color="black")
ax[1].axhline(0, linestyle="--", color="black")
plt.savefig("suppfig_contra_ipsi_traj.pdf", transparent=true, bbox_inches="tight")
    
fig, ax = plt.subplots(nrows=1, ncols=2, width_ratios=[0.1, 1.9], figsize=(5,4))
xt = 1:15
ax[1].plot([0], [nacc_ipsi_contra.cortraj[1]], "-o", lw=3)
ax[1].plot([0], [dms_ipsi_contra.cortraj[1]], "-o", lw=3)
ax[1].plot([0], [dls_ipsi_contra.cortraj[1]], "-o", lw=3)
ax[2].plot(xt, nacc_ipsi_contra.cortraj[2:16], "-o", lw=3)
ax[2].plot(xt, dms_ipsi_contra.cortraj[2:16], "-o", lw=3)
ax[2].plot(xt, dls_ipsi_contra.cortraj[2:16], "-o", lw=3)
ax[2].spines["left"].set_visible(false)
ax[2].set_yticks([])
ax[1].set_ylim(-0.7, 1.)
ax[2].set_ylim(-0.7, 1.)
ax[1].set_xlim(-0.01, 0.01)
ax[1].set_xticks([0])
ax[1].set_yticks([-0.5, 0, 0.5, 1])
plt.subplots_adjust(wspace=0.07)

ax[1].annotate(
    xy = [0, 0.8], text = pval_stars(nacc_ipsi_contra.pvaltraj[1]), rotation=90, color="tab:blue"
)

ax[1].annotate(
    xy = [0, 0.9], text = pval_stars(dms_ipsi_contra.pvaltraj[1]), rotation=90, color="tab:orange"
)

ax[1].annotate(
    xy = [0, 1], text = pval_stars(dls_ipsi_contra.pvaltraj[1]), rotation=90, color="tab:green"
)

for (day,j) in enumerate(2:16)
    ax[2].annotate(
        xy = [day-0.3, 0.8], text = pval_stars(nacc_ipsi_contra.pvaltraj[j]), rotation=90, color="tab:blue"
    )

    ax[2].annotate(
        xy = [day-0.3, 0.9], text = pval_stars(dms_ipsi_contra.pvaltraj[j]), rotation=90, color="tab:orange"
    )

    ax[2].annotate(
        xy = [day-0.3, 1], text = pval_stars(dls_ipsi_contra.pvaltraj[j]), rotation=90, color="tab:green"
    )
end

ax[2].set_xticks([1, 5, 10, 15])
ax[1].set_xticks([0])
ax[2].axhline(0, linestyle="--", color="black")
ax[1].axhline(0, linestyle="--", color="black")
plt.savefig("suppfig_ipsi_contra_traj.pdf", transparent=true, bbox_inches="tight")