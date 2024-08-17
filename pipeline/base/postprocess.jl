include(joinpath(@__DIR__, "preprocess.jl"))
include(joinpath(@__DIR__, "constants.jl"))
using JLD2;

# load results
results = load("/jukebox/witten/yoel/saved_results/neural_results.jld2", "results");
kernel_norm = results.kernel_norm;
mouse_ids = [collect(13:16); collect(26:43)];

qc_dict = Dict(
    f => get_qc("/jukebox/witten/ONE/alyx.internationalbrainlab.org/wittenlab/Subjects", f)
    for f in mouse_ids
    )

function make_nan_from_qc(kernel_norm, qc_dict, mouse_ids)
    for mouse in mouse_ids
        for evn in event_names
            for reg in ["NAcc", "DMS", "DLS"]
                qc_days = qc_dict[mouse].session
                qc_reg = qc_dict[mouse][!, Symbol("QC_$(reg)")]
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


# these replicate figures 2 e
contra_dms_corrs = avg_contra_correlation_conmod_dms(kernel_norm, behavior_weight, mouse_ids)
contra_dls_corrs = avg_contra_correlation_conmod_dls(kernel_norm, behavior_weight, mouse_ids)
contra_nacc_corrs = avg_contra_correlation_conmod_nacc(kernel_norm, behavior_weight, mouse_ids)

ipsi_dms_corrs = avg_ipsi_correlation_conmod_dms(kernel_norm, behavior_weight, mouse_ids)
ipsi_dls_corrs = avg_ipsi_correlation_conmod_dls(kernel_norm, behavior_weight, mouse_ids)
ipsi_nacc_corrs = avg_ipsi_correlation_conmod_nacc(kernel_norm, behavior_weight, mouse_ids)



