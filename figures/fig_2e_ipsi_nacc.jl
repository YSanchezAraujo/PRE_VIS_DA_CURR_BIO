include("/Users/ysa/paper_analysis_prereq.jl");


dms_wcidx = argmax(dms_w_cor_c)
dls_wcidx = argmax(dls_w_cor_i)
nacc_wcidx = argmax(nacc_w_cor_c)

ij = 15
dms_wcidx = ij
dls_wcidx = ij
nacc_wcidx = ij

fig, ax = plt.subplots(figsize=(4, 3))
xt = 1:20

neural_vals = ipsi_nacc.cm[xt, nacc_wcidx]
beh_vals = ipsi_nacc.beh[xt, nacc_wcidx]
# compute scaling and shifting factor
model = binary_nanlm(beh_vals, neural_vals)
ax.plot(xt, neural_vals, lw=3, color="tab:blue", linestyle="--")
ax.set_ylim(-2, 13)
axt = ax.twinx()
ax.spines["right"].set_visible(true)
axt.plot(xt, beh_vals, lw=3, linestyle="-", color="tab:blue")
axt.set_ylim((-2-model.b0)/model.beta, (13-model.b0)/model.beta)

ax.set_yticks([0, 5, 10])
ax.set_xticks([1, 10, 20])

ax.tick_params(axis="both", which="major", labelsize=25)
ax.tick_params(axis="both", which="minor", labelsize=25)
axt.tick_params(axis="both", which="minor", labelsize=25)
ax.set_ylabel("neural", fontsize=25)
plt.savefig("ipsi_nacc_beh_neu_cor_v4.pdf", bbox_inches="tight", transparent=true)