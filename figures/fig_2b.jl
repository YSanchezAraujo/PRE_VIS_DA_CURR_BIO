include("/Users/ysa/Documents/da_paper_code/fig/paper_analysis_prereq.jl");
include("/Users/ysa/Documents/da_paper_code/fig/figboilerplate.jl");


day = 6
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 4))
xt = collect(0:49) ./ 49
fip = 27
d1c = neu[fip]["dms"]["sr"][day, :, :]
d1ce = neu[fip]["errdms"]["sr"][day, :, :]
for i in 1:4
    ax.plot(xt, d1c[:, i], lw=3, color=fluo_colors_dms[end:-1:1][i])
    ax.fill_between(xt, 
        d1c[:, i] .- d1ce[:, i], 
        d1c[:, i] .+ d1ce[:, i], 
        alpha = 0.3, color=fluo_colors_dms[end:-1:1][i]
        )
end
#ax.spines["bottom"].set_visible(false)
ax.set_xticks([])
ax.set_yticks([])
#ax.spines["left"].set_visible(false)

ax.set_ylim([-1., 5])
ax.set_xticks([0, 1])
ax.set_yticks([0, 5])
plt.xticks(fontsize=50)
plt.yticks(fontsize=50)

plt.savefig("dms_exp_kern.pdf", bbox_inches="tight", transparent=true)


day = 6
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 4))
xt = collect(0:49) ./ 49
fip = 27
d1c = neu[fip]["nacc"]["sr"][day, :, :]
d1ce = neu[fip]["errnacc"]["sr"][day, :, :]
for i in 1:4
    ax.plot(xt, d1c[:, i], lw=3, color=fluo_colors_nacc[end:-1:1][i])
    ax.fill_between(xt, 
        d1c[:, i] .- d1ce[:, i], 
        d1c[:, i] .+ d1ce[:, i], 
        alpha = 0.3, color=fluo_colors_nacc[end:-1:1][i]
        )
end
#ax.spines["bottom"].set_visible(false)
ax.set_xticks([])
ax.set_yticks([])
#ax.set_ylim([-1., 6])
ax.set_ylim([-1., 5])
ax.set_xticks([0, 1])
ax.set_yticks([0, 5])
plt.xticks(fontsize=50)
plt.yticks(fontsize=50)
#ax.spines["left"].set_visible(false)
plt.savefig("nacc_exp_kern.pdf", bbox_inches="tight", transparent=true)

day = 6
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 4))
xt = collect(0:49) ./ 49
fip = 27
d1c = neu[fip]["dls"]["sr"][day, :, :]
d1ce = neu[fip]["errdls"]["sr"][day, :, :]
for i in 1:4
    ax.plot(xt, d1c[:, i], lw=3, color=fluo_colors_dls[end:-1:1][i])
    ax.fill_between(xt, 
        d1c[:, i] .- d1ce[:, i], 
        d1c[:, i] .+ d1ce[:, i], 
        alpha = 0.3, color=fluo_colors_dls[end:-1:1][i]
        )
end
#ax.spines["bottom"].set_visible(false)
ax.set_ylim([-1., 5])
ax.set_xticks([0, 1])
ax.set_yticks([0, 5])
plt.xticks(fontsize=50)
plt.yticks(fontsize=50)
#ax.spines["left"].set_visible(false)
plt.savefig("dls_exp_kern.pdf", bbox_inches="tight", transparent=true)
