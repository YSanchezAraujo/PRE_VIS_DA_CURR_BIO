include("/Users/ysa/Documents/da_paper_code/fig/paper_analysis_prereq.jl");
include("/Users/ysa/Documents/da_paper_code/fig/figboilerplate.jl");


avg_reg_c = nanmean(contra_dms.all, 3)[1:20, :]
err_reg_c = nansem(contra_dms.all, 3)[1:20, :]

fig, ax = plt.subplots(figsize=(5,4))
xt = 1:20
for i in 1:4
    ax.plot(xt, avg_reg_c[:, i], lw=3, color=fluo_colors_dms[end:-1:1][i], "-o")
    ax.fill_between(xt, avg_reg_c[:, i] .- err_reg_c[:, i], avg_reg_c[:, i] .+ err_reg_c[:, i],
        alpha=0.4, color=fluo_colors_dms[end:-1:1][i])
end
#ax.set_xticks([1, 5, 10, 15, 20])
ax.set_xticks([])
#ax.set_xlabel("training day")
# ax.set_ylabel(
#     latexstring("avg \$| \\boldsymbol{k}_{stimulus}|_2\$")
# )
ax.set_yticks([])
ax.spines["left"].set_visible(false)
#ax.set_title("DMS")
ax.set_ylim(0, 14)
#ax.spines["bottom"].set_visible(false)
ax.tick_params(axis="both", which="major", labelsize=25)
ax.tick_params(axis="both", which="minor", labelsize=25)
# ax.set_ylabel(
#     latexstring("avg \$|| \\boldsymbol{k}_{stimulus} ||_2\$")
# )
ax.set_xlabel("training day")
ax.set_xticks([1, 10, 20])
#plt.savefig("avg_stim_resp_ex.pdf", bbox_inches="tight", transparent=true)
plt.savefig("contra_avg_stim_resp_dms.pdf", bbox_inches="tight", transparent=true)


avg_reg_c = nanmean(ipsi_dms.all, 3)[1:20, :]
err_reg_c = nansem(ipsi_dms.all, 3)[1:20, :]

fig, ax = plt.subplots(figsize=(5, 4))
xt = 1:20
for i in 1:4
    ax.plot(xt, avg_reg_c[:, i], lw=3, color=fluo_colors_dms[end:-1:1][i], "-o")
    ax.fill_between(xt, avg_reg_c[:, i] .- err_reg_c[:, i], avg_reg_c[:, i] .+ err_reg_c[:, i],
        alpha=0.4, color=fluo_colors_dms[end:-1:1][i])
end
#ax.set_xticks([1, 5, 10, 15, 20])
ax.set_xticks([])
ax.set_yticks([])
#ax.set_xlabel("training day")
#ax.set_ylabel(
#    latexstring("avg \$| \\boldsymbol{k}_{stimulus}|_2\$")
#)
#ax.set_title("DMS")
ax.spines["left"].set_visible(false)
ax.set_ylim(0, 14)
#ax.spines["bottom"].set_visible(false)
ax.set_xticks([1, 10, 20])
ax.tick_params(axis="both", which="major", labelsize=25)
ax.tick_params(axis="both", which="minor", labelsize=25)
plt.savefig("ipsi_avg_stim_resp_dms.pdf", bbox_inches="tight", transparent=true)


avg_reg_c = nanmean(contra_dls.all, 3)[1:20, :]
err_reg_c = nansem(contra_dls.all, 3)[1:20, :]

fig, ax = plt.subplots(figsize=(5, 4))
xt = 1:20
for i in 1:4
    ax.plot(xt, avg_reg_c[:, i], lw=3, color=fluo_colors_dls[end:-1:1][i], "-o")
    ax.fill_between(xt, avg_reg_c[:, i] .- err_reg_c[:, i], avg_reg_c[:, i] .+ err_reg_c[:, i],
        alpha=0.4, color=fluo_colors_dls[end:-1:1][i])
end
ax.set_xticks([1, 10, 20])
#ax.set_xlabel("training day")
# ax.set_ylabel(
#     latexstring("avg \$| \\boldsymbol{k}_{stimulus}|_2\$")
# )
ax.set_yticks([])
ax.spines["left"].set_visible(false)
#ax.set_yticks([1, 6, 11])
#ax.set_title("DLS")
ax.set_ylim(0, 14)
ax.tick_params(axis="both", which="major", labelsize=25)
ax.tick_params(axis="both", which="minor", labelsize=25)
plt.savefig("avg_stim_resp_dls.pdf", bbox_inches="tight", transparent=true)

avg_reg_c = nanmean(ipsi_dls.all, 3)[1:20, :]
err_reg_c = nansem(ipsi_dls.all, 3)[1:20, :]

fig, ax = plt.subplots(figsize=(5,4))
xt = 1:20
for i in 1:4
    ax.plot(xt, avg_reg_c[:, i], lw=3, color=fluo_colors_dls[end:-1:1][i], "-o")
    ax.fill_between(xt, avg_reg_c[:, i] .- err_reg_c[:, i], avg_reg_c[:, i] .+ err_reg_c[:, i],
        alpha=0.4, color=fluo_colors_dls[end:-1:1][i])
end
ax.set_xticks([1, 5, 10, 15, 20])
#ax.set_xlabel("training day")
ax.spines["left"].set_visible(false)
ax.set_yticks([])
#ax.set_ylabel(
#    latexstring("avg \$|w_{stimulus}|_2\$")
#)
#ax.set_title("DLS")
ax.set_ylim(0, 14)
ax.set_xticks([1, 10,  20])
ax.tick_params(axis="both", which="major", labelsize=25)
ax.tick_params(axis="both", which="minor", labelsize=25)
plt.savefig("ipsi_avg_stim_resp_dls.pdf", bbox_inches="tight", transparent=true)


avg_reg_c = nanmean(contra_nacc.all, 3)[1:20, :]
err_reg_c = nansem(contra_nacc.all, 3)[1:20, :]

fig, ax = plt.subplots(figsize=(5,4))
xt = 1:20
for i in 1:4
    ax.plot(xt, avg_reg_c[:, i], lw=3, color=fluo_colors_nacc[end:-1:1][i], "-o")
    ax.fill_between(xt, avg_reg_c[:, i] .- err_reg_c[:, i], avg_reg_c[:, i] .+ err_reg_c[:, i],
        alpha=0.4, color=fluo_colors_nacc[end:-1:1][i])
end
#ax.set_xticks([1, 5, 10, 15, 20])
#ax.set_xlabel("training day")
#ax.spines["bottom"].set_visible(false)
ax.set_xticks([])
ax.set_ylabel(
    latexstring("avg \$|| \\boldsymbol{k}_{stimulus}||_2\$")
)
ax.set_yticks([1, 6, 11])
#ax.set_title("NACC")
ax.set_ylim(0, 14)
ax.set_xticks([1, 10,  20])
ax.tick_params(axis="both", which="major", labelsize=25)
ax.tick_params(axis="both", which="minor", labelsize=25)
plt.savefig("avg_stim_resp_nacc.pdf", bbox_inches="tight", transparent=true)


avg_reg_c = nanmean(ipsi_nacc.all, 3)[1:20, :]
err_reg_c = nansem(ipsi_nacc.all, 3)[1:20, :]

fig, ax = plt.subplots(figsize=(5, 4))
xt = 1:20
for i in 1:4
    ax.plot(xt, avg_reg_c[:, i], lw=3, color=fluo_colors_nacc[end:-1:1][i], "-o")
    ax.fill_between(xt, avg_reg_c[:, i] .- err_reg_c[:, i], avg_reg_c[:, i] .+ err_reg_c[:, i],
        alpha=0.4, color=fluo_colors_nacc[end:-1:1][i])
end
#ax.set_xticks([1, 5, 10, 15, 20])
#ax.set_xlabel("training day")
#ax.spines["bottom"].set_visible(false)
ax.set_xticks([])
#ax.set_ylabel(
#    latexstring("avg \$|w_{stimulus}|_2\$")
#)
#ax.spines["bottom"].set_visible(false)
ax.set_xticks([])
#ax.set_title("NACC")
ax.set_ylim(0, 14)
#ax.spines["left"].set_visible(false)
ax.set_yticks([1, 6, 11])
ax.set_xticks([1,  10,  20])
ax.set_ylabel(
    latexstring("avg \$|| \\boldsymbol{k}_{stimulus}||_2\$")
)
ax.tick_params(axis="both", which="major", labelsize=25)
ax.tick_params(axis="both", which="minor", labelsize=25)
plt.savefig("ipsi_avg_stim_resp_nacc.pdf", bbox_inches="tight", transparent=true)

