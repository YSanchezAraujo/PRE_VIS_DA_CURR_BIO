include(joinpath(@__DIR__, "../base/preprocess.jl"));
include(joinpath(@__DIR__, "../base/constants.jl"));
include(joinpath(@__DIR__, "../base/correlation_functions.jl"));
include(joinpath(@__DIR__, "../base/utility.jl"));
include(joinpath(@__DIR__, "../base/figboilerplate.jl"));


mouse_ids = [collect(13:16); collect(26:43)];
n_mice = length(mouse_ids);

# load day0 results
neu_day0_results = load("/jukebox/witten/yoel/saved_results/day0_neural_results.jld2", "results");

day0_mice = 26:43;

function kernel_indiv_avg_sem(list_of_kernels; n_stim=4)
    kern_len = size(list_of_kernels[1], 1)
    n_kern_set = length(list_of_kernels)
    reshape_dims = (kern_len, n_stim, n_kern_set)
    K = reshape(hcat(list_of_kernels...), reshape_dims)
    return (
        avg = nanmean(K, 3), 
        sem = nansem(K, 3),
        indiv = K
    )
end

contra_dms_kernel = kernel_indiv_avg_sem(
    [neu_day0_results.kernel[f]["DMS"][dms_contra_map[f]] for f in day0_mice]
);

ipsi_dms_kernel = kernel_indiv_avg_sem(
    [neu_day0_results.kernel[f]["DMS"][dms_ipsi_map[f]] for f in day0_mice]
);

contra_dls_kernel = kernel_indiv_avg_sem(
    [neu_day0_results.kernel[f]["DLS"][dms_ipsi_map[f]] for f in day0_mice]
);

ipsi_dls_kernel = kernel_indiv_avg_sem(
    [neu_day0_results.kernel[f]["DLS"][dms_contra_map[f]] for f in day0_mice]
);

contra_nacc_kernel = kernel_indiv_avg_sem(
    [neu_day0_results.kernel[f]["NAcc"][dms_ipsi_map[f]] for f in day0_mice]
);

ipsi_nacc_kernel = kernel_indiv_avg_sem(
    [neu_day0_results.kernel[f]["NAcc"][dms_contra_map[f]] for f in day0_mice]
);

xt = collect(0:49) ./ 49
reshape_dims = (50, 4, 18)

fig, ax = plt.subplots()
for i in 1:4
    ax.plot(xt, contra_nacc_kernel.avg[:, i], color = fluo_colors_nacc[end:-1:1][i], alpha = 1)
    ax.fill_between(xt, contra_nacc_kernel.avg[:, i] .- contra_nacc_kernel.sem[:, i],
                        contra_nacc_kernel.avg[:, i] .+ contra_nacc_kernel.sem[:, i],
                        alpha = 0.5, color = fluo_colors_nacc[end:-1:1][i]
    )
end
ax.set_xticks([])
ax.set_yticks([0, 0.5, 1])
plt.savefig("figure_3b_contra_nacc.pdf", format="pdf", transparent=true)

plt.close()
fig, ax = plt.subplots()
for i in 1:4
    ax.plot(xt, ipsi_nacc_kernel.avg[:, i], color = fluo_colors_nacc[end:-1:1][i], alpha = 1)
    ax.fill_between(xt, ipsi_nacc_kernel.avg[:, i] .- ipsi_nacc_kernel.sem[:, i],
                        ipsi_nacc_kernel.avg[:, i] .+ ipsi_nacc_kernel.sem[:, i],
                        alpha = 0.5, color = fluo_colors_nacc[end:-1:1][i]
    )
end
ax.set_xticks([])
ax.set_yticks([0, 0.5, 1])
plt.savefig("figure_3b_ipsi_nacc.pdf", format="pdf", transparent=true)

plt.close()
fig, ax = plt.subplots()
for i in 1:4
    ax.plot(xt, contra_dls_kernel.avg[:, i], color = fluo_colors_dls[end:-1:1][i], alpha = 1)
    ax.fill_between(xt, contra_dls_kernel.avg[:, i] .- contra_dls_kernel.sem[:, i],
                        contra_dls_kernel.avg[:, i] .+ contra_dls_kernel.sem[:, i],
                        alpha = 0.5, color = fluo_colors_dls[end:-1:1][i]
    )
end
ax.set_xticks([0, 1])
ax.set_yticks([0, 0.5, 1])
plt.savefig("figure_3b_contra_dls.pdf", format="pdf", transparent=true)

plt.close()
fig, ax = plt.subplots()
for i in 1:4
    ax.plot(xt, ipsi_dls_kernel.avg[:, i], color = fluo_colors_dls[end:-1:1][i], alpha = 1)
    ax.fill_between(xt, ipsi_dls_kernel.avg[:, i] .- ipsi_dls_kernel.sem[:, i],
                        ipsi_dls_kernel.avg[:, i] .+ ipsi_dls_kernel.sem[:, i],
                        alpha = 0.5, color = fluo_colors_dls[end:-1:1][i]
    )
end
ax.set_xticks([0, 1])
ax.set_yticks([0, 0.5, 1])
plt.savefig("figure_3b_ipsi_dls.pdf", format="pdf", transparent=true)

plt.close()
fig, ax = plt.subplots()
for i in 1:4
    ax.plot(xt, contra_dms_kernel.avg[:, i], color = fluo_colors_dms[end:-1:1][i], alpha = 1)
    ax.fill_between(xt, contra_dms_kernel.avg[:, i] .- contra_dms_kernel.sem[:, i],
                        contra_dms_kernel.avg[:, i] .+ contra_dms_kernel.sem[:, i],
                        alpha = 0.5, color = fluo_colors_dms[end:-1:1][i]
    )
end
ax.set_xticks([0, 1])
ax.set_yticks([0, 0.5, 1])
plt.savefig("figure_3b_contra_dms.pdf", format="pdf", transparent=true)

plt.close()
fig, ax = plt.subplots()
for i in 1:4
    ax.plot(xt, ipsi_dms_kernel.avg[:, i], color = fluo_colors_dms[end:-1:1][i], alpha = 1)
    ax.fill_between(xt, ipsi_dms_kernel.avg[:, i] .- ipsi_dms_kernel.sem[:, i],
                        ipsi_dms_kernel.avg[:, i] .+ ipsi_dms_kernel.sem[:, i],
                        alpha = 0.5, color = fluo_colors_dms[end:-1:1][i]
    )
end
ax.set_xticks([0, 1])
ax.set_yticks([0, 0.5, 1])
plt.savefig("figure_3b_ipsi_dms.pdf", format="pdf", transparent=true)

