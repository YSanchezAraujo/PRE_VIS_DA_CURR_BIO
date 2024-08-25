include(joinpath(@__DIR__, "../base/preprocess.jl"));
include(joinpath(@__DIR__, "../base/constants.jl"));
include(joinpath(@__DIR__, "../base/correlation_functions.jl"));
include(joinpath(@__DIR__, "../base/utility.jl"));
include(joinpath(@__DIR__, "../base/figboilerplate.jl"));

neu_results = load("/jukebox/witten/yoel/saved_results/neural_results.jld2", "results");
day = 6
mouse = 27

# DMS FIGURE
d1c = neu_results.kernel[mouse]["DMS"]["stim_right"][day, :, :]
d1ce = neu_results.error[mouse]["DMS"]["stim_right"][day, :, :]
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 4))
xt = collect(0:49) ./ 49
for i in 1:4
    ax.plot(xt, d1c[:, i], lw=3, color=fluo_colors_dms[end:-1:1][i])
    ax.fill_between(xt, 
        d1c[:, i] .- d1ce[:, i], 
        d1c[:, i] .+ d1ce[:, i], 
        alpha = 0.3, color=fluo_colors_dms[end:-1:1][i]
        )
end
ax.set_xticks([])
ax.set_yticks([])
ax.set_ylim([-1., 5])
ax.set_xticks([0, 1])
ax.set_yticks([0, 5])
plt.xticks(fontsize=50)
plt.yticks(fontsize=50)
plt.savefig("dms_exp_kern.pdf", bbox_inches="tight", transparent=true)

# NAcc FIGURE
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 4))
xt = collect(0:49) ./ 49
d1c = neu_results.kernel[mouse]["NAcc"]["stim_right"][day, :, :]
d1ce = neu_results.error[mouse]["NAcc"]["stim_right"][day, :, :]
for i in 1:4
    ax.plot(xt, d1c[:, i], lw=3, color=fluo_colors_nacc[end:-1:1][i])
    ax.fill_between(xt, 
        d1c[:, i] .- d1ce[:, i], 
        d1c[:, i] .+ d1ce[:, i], 
        alpha = 0.3, color=fluo_colors_nacc[end:-1:1][i]
        )
end
ax.set_xticks([])
ax.set_yticks([])
ax.set_ylim([-1., 5])
ax.set_xticks([0, 1])
ax.set_yticks([0, 5])
plt.xticks(fontsize=50)
plt.yticks(fontsize=50)
plt.savefig("nacc_exp_kern.pdf", bbox_inches="tight", transparent=true)

# DLS FIGURE
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 4))
xt = collect(0:49) ./ 49
d1c = neu_results.kernel[mouse]["DLS"]["stim_right"][day, :, :]
d1ce = neu_results.error[mouse]["DLS"]["stim_right"][day, :, :]
for i in 1:4
    ax.plot(xt, d1c[:, i], lw=3, color=fluo_colors_dls[end:-1:1][i])
    ax.fill_between(xt, 
        d1c[:, i] .- d1ce[:, i], 
        d1c[:, i] .+ d1ce[:, i], 
        alpha = 0.3, color=fluo_colors_dls[end:-1:1][i]
        )
end
ax.set_ylim([-1., 5])
ax.set_xticks([0, 1])
ax.set_yticks([0, 5])
plt.xticks(fontsize=50)
plt.yticks(fontsize=50)
plt.savefig("dls_exp_kern.pdf", bbox_inches="tight", transparent=true)