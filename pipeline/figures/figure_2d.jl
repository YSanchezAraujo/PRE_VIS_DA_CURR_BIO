include(joinpath(@__DIR__, "../base/preprocess.jl"));
include(joinpath(@__DIR__, "../base/constants.jl"));
include(joinpath(@__DIR__, "../base/correlation_functions.jl"));
include(joinpath(@__DIR__, "../base/utility.jl"));
include(joinpath(@__DIR__, "../base/figboilerplate.jl"));

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

n_days = 20
n_sim = 4
n_mice = 22
contra_stim_dms = [kernel_norm[mouse]["DMS"][dms_contra_map[mouse]] for mouse in mouse_ids];
contra_stim_dms = reshape(hcat(contra_stim_dms...), n_days, n_sim, n_mice);
avg_contra_dms = nanmean(contra_stim_dms, 3);
err_contra_dms = nansem(contra_stim_dms, 3);

ipsi_stim_dms = [kernel_norm[mouse]["DMS"][dms_ipsi_map[mouse]] for mouse in mouse_ids];
ipsi_stim_dms = reshape(hcat(ipsi_stim_dms...), n_days, n_sim, n_mice);
avg_ipsi_dms = nanmean(ipsi_stim_dms, 3);
err_ipsi_dms = nansem(ipsi_stim_dms, 3);

# need to do the same as above for DLS and NAcc
contra_stim_dls = [kernel_norm[mouse]["DLS"][dms_ipsi_map[mouse]] for mouse in mouse_ids];
contra_stim_dls = reshape(hcat(contra_stim_dls...), n_days, n_sim, n_mice);
avg_contra_dls = nanmean(contra_stim_dls, 3);
err_contra_dls = nansem(contra_stim_dls, 3);

ipsi_stim_dls = [kernel_norm[mouse]["DLS"][dms_contra_map[mouse]] for mouse in mouse_ids];
ipsi_stim_dls = reshape(hcat(ipsi_stim_dls...), n_days, n_sim, n_mice);
avg_ipsi_dls = nanmean(ipsi_stim_dls, 3);
err_ipsi_dls = nansem(ipsi_stim_dls, 3);

contra_stim_nacc = [kernel_norm[mouse]["NAcc"][dms_ipsi_map[mouse]] for mouse in mouse_ids];
contra_stim_nacc = reshape(hcat(contra_stim_nacc...), n_days, n_sim, n_mice);
avg_contra_nacc = nanmean(contra_stim_nacc, 3);
err_contra_nacc = nansem(contra_stim_nacc, 3);

ipsi_stim_nacc = [kernel_norm[mouse]["NAcc"][dms_contra_map[mouse]] for mouse in mouse_ids];
ipsi_stim_nacc = reshape(hcat(ipsi_stim_nacc...), n_days, n_sim, n_mice);
avg_ipsi_nacc = nanmean(ipsi_stim_nacc, 3);
err_ipsi_nacc = nansem(ipsi_stim_nacc, 3);

# contra DMS
fig, ax = plt.subplots(figsize=(5,4))
xt = 1:20
for i in 1:4
    ax.plot(xt, avg_contra_dms[:, i], lw=3, color=fluo_colors_dms[end:-1:1][i], "-o")
    ax.fill_between(xt, avg_contra_dms[:, i] .- err_contra_dms[:, i], 
                        avg_contra_dms[:, i] .+ err_contra_dms[:, i],
        alpha=0.4, color=fluo_colors_dms[end:-1:1][i])
end
ax.set_xticks([])
ax.set_yticks([])
ax.spines["left"].set_visible(false)
ax.set_ylim(0, 15)
ax.tick_params(axis="both", which="major", labelsize=25)
ax.tick_params(axis="both", which="minor", labelsize=25)
ax.set_xticks([1, 10, 20])
plt.savefig("contra_avg_stim_resp_dms.pdf", bbox_inches="tight", transparent=true)

# ipsi DMS
fig, ax = plt.subplots(figsize=(5, 4))
xt = 1:20
for i in 1:4
    ax.plot(xt, avg_ipsi_dms[:, i], lw=3, color=fluo_colors_dms[end:-1:1][i], "-o")
    ax.fill_between(xt, avg_ipsi_dms[:, i] .- err_ipsi_dms[:, i], 
                        avg_ipsi_dms[:, i] .+ err_ipsi_dms[:, i],
        alpha=0.4, color=fluo_colors_dms[end:-1:1][i])
end
ax.set_xticks([])
ax.set_yticks([])
ax.spines["left"].set_visible(false)
ax.set_ylim(0, 15)
ax.set_xticks([1, 10, 20])
ax.tick_params(axis="both", which="major", labelsize=25)
ax.tick_params(axis="both", which="minor", labelsize=25)
plt.savefig("ipsi_avg_stim_resp_dms.pdf", bbox_inches="tight", transparent=true)

# contra DLS
fig, ax = plt.subplots(figsize=(5, 4))
xt = 1:20
for i in 1:4
    ax.plot(xt, avg_contra_dls[:, i], lw=3, color=fluo_colors_dls[end:-1:1][i], "-o")
    ax.fill_between(xt, avg_contra_dls[:, i] .- err_contra_dls[:, i], 
                        avg_contra_dls[:, i] .+ err_contra_dls[:, i],
        alpha=0.4, color=fluo_colors_dls[end:-1:1][i])
end
ax.set_xticks([1, 10, 20])
ax.set_yticks([])
ax.spines["left"].set_visible(false)
ax.set_ylim(0, 15)
ax.tick_params(axis="both", which="major", labelsize=25)
ax.tick_params(axis="both", which="minor", labelsize=25)
plt.savefig("contra_avg_stim_resp_dls.pdf", bbox_inches="tight", transparent=true)

# IPSI DLS
fig, ax = plt.subplots(figsize=(5,4))
xt = 1:20
for i in 1:4
    ax.plot(xt, avg_ipsi_dls[:, i], lw=3, color=fluo_colors_dls[end:-1:1][i], "-o")
    ax.fill_between(xt, avg_ipsi_dls[:, i] .- err_ipsi_dls[:, i], 
                        avg_ipsi_dls[:, i] .+ err_ipsi_dls[:, i],
        alpha=0.4, color=fluo_colors_dls[end:-1:1][i])
end
ax.set_xticks([1, 5, 10, 15, 20])
ax.spines["left"].set_visible(false)
ax.set_yticks([])
ax.set_ylim(0, 15)
ax.set_xticks([1, 10,  20])
ax.tick_params(axis="both", which="major", labelsize=25)
ax.tick_params(axis="both", which="minor", labelsize=25)
plt.savefig("ipsi_avg_stim_resp_dls.pdf", bbox_inches="tight", transparent=true)

# CONTRA NAcc
fig, ax = plt.subplots(figsize=(5,4))
xt = 1:20
for i in 1:4
    ax.plot(xt, avg_contra_nacc[:, i], lw=3, color=fluo_colors_nacc[end:-1:1][i], "-o")
    ax.fill_between(xt, avg_contra_nacc[:, i] .- err_contra_nacc[:, i], 
                        avg_contra_nacc[:, i] .+ err_contra_nacc[:, i],
        alpha=0.4, color=fluo_colors_nacc[end:-1:1][i])
end
ax.set_xticks([])
ax.set_ylabel(
    latexstring("avg \$|| \\boldsymbol{k}_{stimulus}||_2\$")
)
ax.set_yticks([1, 6, 11])
ax.set_ylim(0, 15)
ax.set_xticks([1, 10,  20])
ax.tick_params(axis="both", which="major", labelsize=25)
ax.tick_params(axis="both", which="minor", labelsize=25)
plt.savefig("contra_avg_stim_resp_nacc.pdf", bbox_inches="tight", transparent=true)


# IPSI NAcc
fig, ax = plt.subplots(figsize=(5,4))
xt = 1:20
for i in 1:4
    ax.plot(xt, avg_ipsi_nacc[:, i], lw=3, color=fluo_colors_nacc[end:-1:1][i], "-o")
    ax.fill_between(xt, avg_ipsi_nacc[:, i] .- err_ipsi_nacc[:, i], 
                        avg_ipsi_nacc[:, i] .+ err_ipsi_nacc[:, i],
        alpha=0.4, color=fluo_colors_nacc[end:-1:1][i])
end
ax.set_xticks([])
ax.set_ylabel(
    latexstring("avg \$|| \\boldsymbol{k}_{stimulus}||_2\$")
)
ax.set_yticks([1, 6, 11])
ax.set_ylim(0, 15)
ax.set_xticks([1, 10,  20])
ax.tick_params(axis="both", which="major", labelsize=25)
ax.tick_params(axis="both", which="minor", labelsize=25)
plt.savefig("ipsi_avg_stim_resp_nacc.pdf", bbox_inches="tight", transparent=true)