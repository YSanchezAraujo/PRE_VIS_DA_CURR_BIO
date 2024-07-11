include("/Users/ysa/Documents/da_paper_code/fig/paper_analysis_prereq.jl");
include("/Users/ysa/Documents/da_paper_code/fig/figboilerplate.jl");


function compute_corr_trajs(day0_neu, neu_mat, beh_vec, n_day, to_boot, n_boot)

    robust_stats = (
        p = fill(NaN, n_day+1),
        cor = fill(NaN, n_day+1),
        r2 = fill(NaN, n_day+1),
        boot_cor = fill(NaN, n_day+1, n_boot),
        boot_p = fill(NaN, n_day+1, n_boot),
        boot_r2 = fill(NaN, n_day+1, n_boot)
    )

    linear_stats = (
        p = fill(NaN, n_day+1),
        cor = fill(NaN, n_day+1),
        r2 = fill(NaN, n_day+1),
        boot_cor = fill(NaN, n_day+1, n_boot),
        boot_p = fill(NaN, n_day+1, n_boot),
        boot_r2 = fill(NaN, n_day+1, n_boot)
    )

    n_mouse = length(beh_vec)
    mouse_set = 1:1:n_mouse

    day0_robust_model = binary_nan_robust_lm(day0_neu, beh_vec)
    day0_linear_model = binary_nan_lm(day0_neu, beh_vec)

    linear_stats.p[1] = day0_linear_model.p
    linear_stats.cor[1] = day0_linear_model.cor
    linear_stats.r2[1] = day0_linear_model.r2

    robust_stats.p[1] = day0_robust_model.p
    robust_stats.cor[1] = day0_robust_model.cor
    robust_stats.r2[1] = day0_robust_model.r2 

    for boot_idx in 1:n_boot
        boot_inds = sample(mouse_set, n_mouse, replace=true)
        # run models
        day0_boot_robust_model = binary_nan_robust_lm(day0_neu[boot_inds], beh_vec)
        day0_boot_linear_model = binary_nan_lm(day0_neu[boot_inds], beh_vec)
        # save results linear model
        linear_stats.boot_p[1, boot_idx] = day0_boot_linear_model.p
        linear_stats.boot_cor[1, boot_idx] = day0_boot_linear_model.cor
        linear_stats.boot_r2[1, boot_idx] = day0_boot_linear_model.r2
        # robust model
        robust_stats.boot_p[1, boot_idx] = day0_boot_robust_model.p
        robust_stats.boot_cor[1, boot_idx] = day0_boot_robust_model.cor
        robust_stats.boot_r2[1, boot_idx] = day0_boot_robust_model.r2 
    end

    for day in 1:n_day
        day_robust_model = binary_nan_robust_lm(neu_mat[day, :], beh_vec)
        day_linear_model = binary_nan_lm(neu_mat[day, :], beh_vec)

        linear_stats.p[day+1] = day_linear_model.p
        linear_stats.cor[day+1] = day_linear_model.cor
        linear_stats.r2[day+1] = day_linear_model.r2

        robust_stats.p[day+1] = day_robust_model.p
        robust_stats.cor[day+1] = day_robust_model.cor
        robust_stats.r2[day+1] = day_robust_model.r2 

        for boot_idx in 1:n_boot
            boot_inds = sample(mouse_set, n_mouse, replace=true)
            try
                # run models
                boot_robust_model = binary_nan_robust_lm(neu_mat[day, :][boot_inds], beh_vec)
                boot_linear_model = binary_nan_lm(neu_mat[day, :][boot_inds], beh_vec)
                # save results linear model
                linear_stats.boot_p[day+1, boot_idx] = boot_linear_model.p
                linear_stats.boot_cor[day+1, boot_idx] = boot_linear_model.cor
                linear_stats.boot_r2[day+1, boot_idx] = boot_linear_model.r2
                # robust model
                robust_stats.boot_p[day+1, boot_idx] = boot_robust_model.p
                robust_stats.boot_cor[day+1, boot_idx] = boot_robust_model.cor
                robust_stats.boot_r2[day+1, boot_idx] = boot_robust_model.r2 
            catch
                println(string("bootstrap ", boot_idx, " errored out"))
            end
        end
    end

    return (linear=linear_stats, robust=robust_stats)

end


mt = 5:22
to_boot = false
n_boot = 1
corr_set = (
    nacc = (
        contra_contra = compute_corr_trajs(day0_contra_nacc.cm, contra_nacc.cm[:, mt], nanmean(contra_nacc.beh[end-4:end, mt], 1), 15, to_boot, n_boot),
        contra_ipsi = compute_corr_trajs(day0_contra_nacc.cm, contra_nacc.cm[:, mt], nanmean(ipsi_nacc.beh[end-4:end, mt], 1), 15, to_boot, n_boot),
        ipsi_ipsi = compute_corr_trajs(day0_ipsi_nacc.cm, ipsi_nacc.cm[:, mt], nanmean(ipsi_nacc.beh[end-4:end, mt], 1), 15, to_boot, n_boot),
        ipsi_contra = compute_corr_trajs(day0_ipsi_nacc.cm, ipsi_nacc.cm[:, mt], nanmean(contra_nacc.beh[end-4:end, mt], 1), 15, to_boot, n_boot)

        ),
    dms = (
        contra_contra = compute_corr_trajs(day0_contra_dms.cm, contra_dms.cm[:, mt], nanmean(contra_dms.beh[end-4:end, mt], 1), 15, to_boot, n_boot),
        contra_ipsi = compute_corr_trajs(day0_contra_dms.cm, contra_dms.cm[:, mt], nanmean(ipsi_dms.beh[end-4:end, mt], 1), 15, to_boot, n_boot),
        ipsi_ipsi = compute_corr_trajs(day0_ipsi_dms.cm, ipsi_dms.cm[:, mt], nanmean(ipsi_dms.beh[end-4:end, mt], 1), 15, to_boot, n_boot),
        ipsi_contra = compute_corr_trajs(day0_ipsi_dms.cm, ipsi_dms.cm[:, mt], nanmean(contra_dms.beh[end-4:end, mt], 1), 15, to_boot, n_boot)

        ),
    dls = (
        contra_contra = compute_corr_trajs(day0_contra_dls.cm, contra_dls.cm[:, mt], nanmean(contra_dls.beh[end-4:end, mt], 1), 15, to_boot, n_boot),
        contra_ipsi = compute_corr_trajs(day0_contra_dls.cm, contra_dls.cm[:, mt], nanmean(ipsi_dls.beh[end-4:end, mt], 1), 15, to_boot, n_boot),
        ipsi_ipsi = compute_corr_trajs(day0_ipsi_dls.cm, ipsi_dls.cm[:, mt], nanmean(ipsi_dls.beh[end-4:end, mt], 1), 15, to_boot, n_boot),
        ipsi_contra = compute_corr_trajs(day0_ipsi_dls.cm, ipsi_dls.cm[:, mt], nanmean(contra_dls.beh[end-4:end, mt], 1), 15, to_boot, n_boot)
        ),
);






fig, ax = plt.subplots(nrows=1, ncols=2, width_ratios=[0.1, 1.9], figsize=(5,4))
xt = 1:15
ax[1].plot([0], [corr_set.nacc.contra_contra.robust.cor[1]], "-o", lw=3)
ax[1].plot([0], [corr_set.dms.contra_contra.robust.cor[1]], "-o", lw=3)
ax[1].plot([0], [corr_set.dls.contra_contra.robust.cor[1]], "-o", lw=3)
ax[2].plot(xt, corr_set.nacc.contra_contra.robust.cor[2:16], "-o", lw=3)
ax[2].plot(xt, corr_set.dms.contra_contra.robust.cor[2:16], "-o", lw=3)
ax[2].plot(xt, corr_set.dls.contra_contra.robust.cor[2:16], "-o", lw=3)
ax[2].spines["left"].set_visible(false)
ax[2].set_yticks([])
ax[1].set_ylim(-0.7, 1.)
ax[2].set_ylim(-0.7, 1.)
ax[1].set_xlim(-0.01, 0.01)
ax[1].set_xticks([0])
ax[1].set_yticks([-0.5, 0, 0.5, 1])
plt.subplots_adjust(wspace=0.07)


ax[1].annotate(
    xy = [0, 0.8], text = pval_stars(corr_set.nacc.contra_contra.robust.p[1]), rotation=90, color="tab:blue"
)

ax[1].annotate(
    xy = [0, 0.9], text = pval_stars(corr_set.dms.contra_contra.robust.p[1]), rotation=90, color="tab:orange"
)

ax[1].annotate(
    xy = [0, 1], text = pval_stars(corr_set.dls.contra_contra.robust.p[1]), rotation=90, color="tab:green"
)

for (day,j) in enumerate(2:16)
    ax[2].annotate(
        xy = [day-0.3, 0.8], text = pval_stars(corr_set.nacc.contra_contra.robust.p[j]), rotation=90, color="tab:blue"
    )

    ax[2].annotate(
        xy = [day-0.3, 0.9], text = pval_stars(corr_set.dms.contra_contra.robust.p[j]), rotation=90, color="tab:orange"
    )

    ax[2].annotate(
        xy = [day-0.3, 1], text = pval_stars(corr_set.dls.contra_contra.robust.p[j]), rotation=90, color="tab:green"
    )
end

ax[2].set_xticks([1, 15])
ax[1].set_xticks([0])
#ax[2].set_xlabel("contra neural on day \n v.s. late contra behavior")
#ax[2].set_title("contra behavior days 16 to 20", fontsize=20)
#ax[1].set_ylabel("correlation")
ax[2].axhline(0, linestyle="--", color="black")
ax[1].axhline(0, linestyle="--", color="black")

plt.savefig("contra_contra_cortrajs_v2.pdf", bbox_inches="tight", transparent=true)





######################################################################################################
######################################################################################################
######################################################################################################



fig, ax = plt.subplots(nrows=1, ncols=2, width_ratios=[0.1, 1.9], figsize=(5,4))
xt = 1:15
ax[1].plot([0], [corr_set.nacc.contra_ipsi.robust.cor[1]], "-o", lw=3)
ax[1].plot([0], [corr_set.dms.contra_ipsi.robust.cor[1]], "-o", lw=3)
ax[1].plot([0], [corr_set.dls.contra_ipsi.robust.cor[1]], "-o", lw=3)
ax[2].plot(xt, corr_set.nacc.contra_ipsi.robust.cor[2:16], "-o", lw=3)
ax[2].plot(xt, corr_set.dms.contra_ipsi.robust.cor[2:16], "-o", lw=3)
ax[2].plot(xt, corr_set.dls.contra_ipsi.robust.cor[2:16], "-o", lw=3)
ax[2].spines["left"].set_visible(false)
ax[2].set_yticks([])
ax[1].set_ylim(-0.7, 1.)
ax[2].set_ylim(-0.7, 1.)
ax[1].set_xlim(-0.01, 0.01)
ax[1].set_xticks([0])
ax[1].set_yticks([-0.5, 0, 0.5, 1])
plt.subplots_adjust(wspace=0.07)



ax[1].annotate(
    xy = [0, 0.8], text = pval_stars(corr_set.nacc.contra_ipsi.robust.p[1]), rotation=90, color="tab:blue"
)

ax[1].annotate(
    xy = [0, 0.9], text = pval_stars(corr_set.dms.contra_ipsi.robust.p[1]), rotation=90, color="tab:orange"
)

ax[1].annotate(
    xy = [0, 1], text = pval_stars(corr_set.dls.contra_ipsi.robust.p[1]), rotation=90, color="tab:green"
)

for (day,j) in enumerate(2:16)
    ax[2].annotate(
        xy = [day-0.3, 0.8], text = pval_stars(corr_set.nacc.contra_ipsi.robust.p[j]), rotation=90, color="tab:blue"
    )

    ax[2].annotate(
        xy = [day-0.3, 0.9], text = pval_stars(corr_set.dms.contra_ipsi.robust.p[j]), rotation=90, color="tab:orange"
    )

    ax[2].annotate(
        xy = [day-0.3, 1], text = pval_stars(corr_set.dls.contra_ipsi.robust.p[j]), rotation=90, color="tab:green"
    )
end

ax[2].set_xticks([1, 15])
ax[1].set_xticks([0])
#ax[2].set_xlabel("contra neural on day \n v.s. late contra behavior")
#ax[2].set_title("contra behavior days 16 to 20", fontsize=20)
#ax[1].set_ylabel("correlation")
ax[2].axhline(0, linestyle="--", color="black")
ax[1].axhline(0, linestyle="--", color="black")

plt.savefig("contra_ipsi_cortrajs_v2.pdf", bbox_inches="tight", transparent=true)





######################################################################################################
######################################################################################################
######################################################################################################


fig, ax = plt.subplots(nrows=1, ncols=2, width_ratios=[0.1, 1.9], figsize=(5,4))
xt = 1:15
ax[1].plot([0], [corr_set.nacc.ipsi_ipsi.robust.cor[1]], "-o", lw=3)
ax[1].plot([0], [corr_set.dms.ipsi_ipsi.robust.cor[1]], "-o", lw=3)
ax[1].plot([0], [corr_set.dls.ipsi_ipsi.robust.cor[1]], "-o", lw=3)
ax[2].plot(xt, corr_set.nacc.ipsi_ipsi.robust.cor[2:16], "-o", lw=3)
ax[2].plot(xt, corr_set.dms.ipsi_ipsi.robust.cor[2:16], "-o", lw=3)
ax[2].plot(xt, corr_set.dls.ipsi_ipsi.robust.cor[2:16], "-o", lw=3)
ax[2].spines["left"].set_visible(false)
ax[2].set_yticks([])
ax[1].set_ylim(-0.7, 1.)
ax[2].set_ylim(-0.7, 1.)
ax[1].set_xlim(-0.01, 0.01)
ax[1].set_xticks([0])
ax[1].set_yticks([-0.5, 0, 0.5, 1])
plt.subplots_adjust(wspace=0.07)




ax[1].annotate(
    xy = [0, 0.8], text = pval_stars(corr_set.nacc.ipsi_ipsi.robust.p[1]), rotation=90, color="tab:blue"
)

ax[1].annotate(
    xy = [0, 0.9], text = pval_stars(corr_set.dms.ipsi_ipsi.robust.p[1]), rotation=90, color="tab:orange"
)

ax[1].annotate(
    xy = [0, 1], text = pval_stars(corr_set.dls.ipsi_ipsi.robust.p[1]), rotation=90, color="tab:green"
)

for (day,j) in enumerate(2:16)
    ax[2].annotate(
        xy = [day-0.3, 0.8], text = pval_stars(corr_set.nacc.ipsi_ipsi.robust.p[j]), rotation=90, color="tab:blue"
    )

    ax[2].annotate(
        xy = [day-0.3, 0.9], text = pval_stars(corr_set.dms.ipsi_ipsi.robust.p[j]), rotation=90, color="tab:orange"
    )

    ax[2].annotate(
        xy = [day-0.3, 1], text = pval_stars(corr_set.dls.ipsi_ipsi.robust.p[j]), rotation=90, color="tab:green"
    )
end

ax[2].set_xticks([1, 15])
ax[1].set_xticks([0])
#ax[2].set_xlabel("contra neural on day \n v.s. late contra behavior")
#ax[2].set_title("contra behavior days 16 to 20", fontsize=20)
#ax[1].set_ylabel("correlation")
ax[2].axhline(0, linestyle="--", color="black")
ax[1].axhline(0, linestyle="--", color="black")

plt.savefig("ipsi_ipsi_cortrajs_v2.pdf", bbox_inches="tight", transparent=true)


######################################################################################################
######################################################################################################
######################################################################################################



fig, ax = plt.subplots(nrows=1, ncols=2, width_ratios=[0.1, 1.9], figsize=(5,4))
xt = 1:15
ax[1].plot([0], [corr_set.nacc.ipsi_contra.robust.cor[1]], "-o", lw=3)
ax[1].plot([0], [corr_set.dms.ipsi_contra.robust.cor[1]], "-o", lw=3)
ax[1].plot([0], [corr_set.dls.ipsi_contra.robust.cor[1]], "-o", lw=3)
ax[2].plot(xt, corr_set.nacc.ipsi_contra.robust.cor[2:16], "-o", lw=3)
ax[2].plot(xt, corr_set.dms.ipsi_contra.robust.cor[2:16], "-o", lw=3)
ax[2].plot(xt, corr_set.dls.ipsi_contra.robust.cor[2:16], "-o", lw=3)
ax[2].spines["left"].set_visible(false)
ax[2].set_yticks([])
ax[1].set_ylim(-0.7, 1.)
ax[2].set_ylim(-0.7, 1.)
ax[1].set_xlim(-0.01, 0.01)
ax[1].set_xticks([0])
ax[1].set_yticks([-0.5, 0, 0.5, 1])
plt.subplots_adjust(wspace=0.07)


ax[1].annotate(
    xy = [0, 0.8], text = pval_stars(corr_set.nacc.ipsi_contra.robust.p[1]), rotation=90, color="tab:blue"
)

ax[1].annotate(
    xy = [0, 0.9], text = pval_stars(corr_set.dms.ipsi_contra.robust.p[1]), rotation=90, color="tab:orange"
)

ax[1].annotate(
    xy = [0, 1], text = pval_stars(corr_set.dls.ipsi_contra.robust.p[1]), rotation=90, color="tab:green"
)

for (day,j) in enumerate(2:16)
    ax[2].annotate(
        xy = [day-0.3, 0.8], text = pval_stars(corr_set.nacc.ipsi_contra.robust.p[j]), rotation=90, color="tab:blue"
    )

    ax[2].annotate(
        xy = [day-0.3, 0.9], text = pval_stars(corr_set.dms.ipsi_contra.robust.p[j]), rotation=90, color="tab:orange"
    )

    ax[2].annotate(
        xy = [day-0.3, 1], text = pval_stars(corr_set.dls.ipsi_contra.robust.p[j]), rotation=90, color="tab:green"
    )
end

ax[2].set_xticks([1, 15])
ax[1].set_xticks([0])
#ax[2].set_xlabel("contra neural on day \n v.s. late contra behavior")
#ax[2].set_title("contra behavior days 16 to 20", fontsize=20)
#ax[1].set_ylabel("correlation")
ax[2].axhline(0, linestyle="--", color="black")
ax[1].axhline(0, linestyle="--", color="black")

plt.savefig("ipsi_contra_cortrajs_v2.pdf", bbox_inches="tight", transparent=true)


