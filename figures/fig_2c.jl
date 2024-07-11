include("/Users/ysa/Documents/da_paper_code/fig/paper_analysis_prereq.jl");
include("/Users/ysa/Documents/da_paper_code/fig/figboilerplate.jl");

preddata = load("/Users/ysa/Documents/da_paper_code/data/data2_for_preds_paperfig2.jld2", "data");

trs = [
    preddata.data_trunc.cue_idx[j]:preddata.data_trunc.fb_idx[j]+49
    for j in [ 150, 200, 230, 235, 240]
]

fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(10, 2))

for i in 1:5
    ax[i].plot(preddata.Y[trs[i], 1], lw=3, color="tab:blue")
    ax[i].plot(preddata.YH[trs[i], 1], linestyle="--", color="black", lw=3)
end
for i in 1:5
    ax[i].spines["left"].set_visible(false)
    ax[i].spines["bottom"].set_visible(false)
    ax[i].set_xticks([])
    ax[i].set_yticks([])
    ax[i].set_ylim(-1, 7)
end
r2ex = round(rsquared(preddata.Y[:, 1], preddata.YH[:, 1]); digits=2)
ax[3].set_title(latexstring("\$R^2: $r2ex\$"), fontsize=23)
plt.savefig("nacc_preds.pdf", bbox_inches="tight", transparent=true)

fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(10, 2))

for i in 1:5
    ax[i].plot(preddata.Y[trs[i],2], lw=3, color="tab:orange")
    ax[i].plot(preddata.YH[trs[i], 2], linestyle="--", color="black", lw=3)
end
for i in 1:5
    ax[i].spines["left"].set_visible(false)
    ax[i].spines["bottom"].set_visible(false)
    ax[i].set_xticks([])
    ax[i].set_yticks([])
    ax[i].set_ylim(-1, 7)
end
r2ex = round(rsquared(preddata.Y[:, 2], preddata.YH[:, 2]); digits=2)
ax[3].set_title(latexstring("\$R^2: $r2ex\$"), fontsize=23)
plt.savefig("dms_preds.pdf", bbox_inches="tight", transparent=true)

fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(10, 2))

for i in 1:5
    ax[i].plot(preddata.Y[trs[i],3], lw=3, color="tab:green")
    ax[i].plot(preddata.YH[trs[i], 3], linestyle="--", color="black", lw=3)
end

for i in 1:5
    ax[i].spines["left"].set_visible(false)
    ax[i].spines["bottom"].set_visible(false)
    ax[i].set_xticks([])
    ax[i].set_yticks([])
    ax[i].set_ylim(-1, 7)
end
r2ex = round(rsquared(preddata.Y[:, 3], preddata.YH[:, 3]); digits=2)
ax[3].set_title(latexstring("\$R^2: $r2ex\$"), fontsize=23)
plt.savefig("dls_preds.pdf", bbox_inches="tight", transparent=true)

