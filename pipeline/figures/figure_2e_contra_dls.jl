include(joinpath(@__DIR__, "../base/preprocess.jl"));
include(joinpath(@__DIR__, "../base/constants.jl"));
include(joinpath(@__DIR__, "../base/correlation_functions.jl"));
include(joinpath(@__DIR__, "../base/utility.jl"));
include(joinpath(@__DIR__, "../base/figboilerplate.jl"));

neu_results = load("/jukebox/witten/yoel/saved_results/neural_results.jld2", "results");
behavior_weight = load("/jukebox/witten/yoel/saved_results/choice_weights.jld2", "results");
kernel_norm = neu_results.kernel_norm;

ij = 15
conmod_norm_contra_dls = hcat([
    (
        kernel_norm[mouse]["DLS"][dms_ipsi_map[mouse]][:, 4] .- 
        kernel_norm[mouse]["DLS"][dms_ipsi_map[mouse]][:, 1] 
    ) 
    for mouse in mouse_ids 
]...)

beh_contra_dls = hcat([
    behavior_weight[mouse].avg[:, beh_dms_ipsi_map[mouse]] for mouse in mouse_ids
]...)

neural_vals = conmod_norm_contra_dls[:, ij]
beh_vals = beh_contra_dls[:, ij]
model = binary_nan_lm(beh_vals, neural_vals; ztransform=false)

fig, ax = plt.subplots(figsize=(4, 3))
xt = 1:20
ax.plot(xt, neural_vals, lw=3, color="tab:green", linestyle="--")
ax.set_ylim(-2, 13)
axt = ax.twinx()
ax.spines["right"].set_visible(true)
axt.plot(xt, beh_vals, lw=3, linestyle="-", color="tab:green")
axt.set_ylim((-2-model.b0)/model.beta, (13-model.b0)/model.beta)
ax.set_yticks([0, 5, 10])
ax.set_xticks([1, 10, 20])
ax.tick_params(axis="both", which="major", labelsize=25)
ax.tick_params(axis="both", which="minor", labelsize=25)
axt.tick_params(axis="both", which="minor", labelsize=25)
axt.set_ylabel("Behavior", fontsize=25)
plt.savefig("contra_dls_beh_neu_cor.pdf", bbox_inches="tight", transparent=true)