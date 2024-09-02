include(joinpath(@__DIR__, "../base/preprocess.jl"));
include(joinpath(@__DIR__, "../base/constants.jl"));
include(joinpath(@__DIR__, "../base/correlation_functions.jl"));
include(joinpath(@__DIR__, "../base/utility.jl"));
include(joinpath(@__DIR__, "../base/figboilerplate.jl"));

neu_results = load("/jukebox/witten/yoel/saved_results/neural_results.jld2", "results");
behavior_weight = load("/jukebox/witten/yoel/saved_results/choice_weights.jld2", "results");
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

ij = 15
conmod_norm_ipsi_dms = hcat([
    (
        kernel_norm[mouse]["DMS"][dms_ipsi_map[mouse]][:, 4] .- 
        kernel_norm[mouse]["DMS"][dms_ipsi_map[mouse]][:, 1] 
    ) 
    for mouse in mouse_ids 
]...)

beh_ipsi_dms = hcat([
    behavior_weight[mouse].avg[:, beh_dms_ipsi_map[mouse]] * beh_multiplier_ipsi[mouse] for mouse in mouse_ids
]...)

neural_vals = conmod_norm_ipsi_dms[:, ij]
beh_vals = beh_ipsi_dms[:, ij]
model = binary_nan_lm(beh_vals, neural_vals; ztransform=false)

fig, ax = plt.subplots(figsize=(4, 3))
xt = 1:20
ax.plot(xt, neural_vals, lw=3, color="tab:orange", linestyle="--")
ax.set_yticks([0, 5, 10])
ax.set_xticks([1, 10, 20])
ax.set_ylim(-2, 13)
axt = ax.twinx()
ax.spines["right"].set_visible(true)
axt.plot(xt, beh_vals, lw=3, linestyle="-", color="tab:orange")
axt.set_ylim((-2-model.b0)/model.beta, (13-model.b0)/model.beta)
ax.tick_params(axis="both", which="major", labelsize=25)
ax.tick_params(axis="both", which="minor", labelsize=25)
axt.tick_params(axis="both", which="minor", labelsize=25)
plt.savefig("ipsi_dms_beh_neu_cor.pdf", bbox_inches="tight", transparent=true)