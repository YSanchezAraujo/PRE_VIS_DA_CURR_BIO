include("/Users/ysa/paper_analysis_prereq.jl");



kmfit = kmeans(day0_contra_dms.cm', 2)

cents = vec(kmfit.centers)
if cents[1] < cents[2]
    strong_dms = kmfit.assignments .== 2
    weak_dms = kmfit.assignments .== 1
else
    strong_dms = kmfit.assignments .== 1
    weak_dms = kmfit.assignments .== 2
end

strong_dms = day0_contra_dms.cm. > median(day0_contra_dms.cm)
weak_dms = .!strong_dms


plt.hist(day0_contra_dms.cm[weak_dms], bins=4,
    color="tab:orange", edgecolor="white", label="weak")

plt.hist(day0_contra_dms.cm[strong_dms], bins=6,
    color="tab:red", edgecolor="white", label="strong")

plt.axvline(2.1, linestyle="--", color="black", lw=3)
plt.xlabel(latexstring("\$ ||\\boldsymbol{K}^{100}_{stimulus} ||_2 -  || \\boldsymbol{K}^{6.25}_{stimulus} ||_2 \$"))
plt.ylabel("counts")
plt.title("Day 0 contra dms")
plt.legend(["weak", "strong"], fontsize=18)
plt.savefig("day0_dms_contra_hist.pdf", bbox_inches="tight", transparent=true)
#############################################################################################
#############################################################################################
#############################################################################################

# LINE PLOT CONTRA
xt = 1:20
mt = 5:22
plt.figure(figsize=(5, 4))
strg_color="tab:red"
d = nanmean(contra_dms.beh[:, mt][:, strong_dms], 2)
plt.plot(xt, d, lw=3, color=strg_color)
v = nansem(contra_dms.beh[:, mt][:, strong_dms], 2) 
plt.fill_between(xt, d .- v, d .+ v, alpha=0.4, color=strg_color, label="_nolegend_")

wk_color="tab:orange"
d = nanmean(contra_dms.beh[:, mt][:, weak_dms], 2)
plt.plot(xt, d, lw=3, color=wk_color)
v = nansem(contra_dms.beh[:, mt][:, weak_dms], 2)
plt.fill_between(xt, d .- v, d .+ v, alpha=0.4, color=wk_color, label="_nolegend_")


plt.xticks([1, 10, 20])
#plt.yticks([0, 5, 8])
# plt.ylim(-2. ,10)
#plt.legend(["strong", "weak"], fontsize=17, loc="upper left")
plt.xlabel("Training day")
plt.ylabel(latexstring("\$\\beta_{contra}\$"))
#plt.text(-4, 12, "F")
plt.savefig("strong_contra_weak_contra_d0_dms_split.pdf", bbox_inches="tight", transparent=true)
#############################################################################################
#############################################################################################
#############################################################################################

# LINE PLOT IPSI
plt.figure(figsize=(5,4))
strg_color="tab:red"
d = nanmean(ipsi_dms.beh[:, mt][:, strong_dms], 2)
plt.plot(xt, d, lw=3, color=strg_color)
v = nansem(ipsi_dms.beh[:, mt][:, strong_dms], 2) 
plt.fill_between(xt, d .- v, d .+ v, alpha=0.4, color=strg_color, label="_nolegend_")

wk_color="tab:orange"
d = nanmean(ipsi_dms.beh[:, mt][:, weak_dms], 2)
plt.plot(xt, d, lw=3, color=wk_color)
v = nansem(ipsi_dms.beh[:, mt][:, weak_dms], 2)
plt.fill_between(xt, d .- v, d .+ v, alpha=0.4, color=wk_color, label="_nolegend_")


plt.xticks([1, 10, 20])
# plt.ylim(-2. ,10)
#plt.legend(["strong", "weak"], fontsize=17, loc="upper left")
plt.xlabel("Training day")
plt.ylabel(latexstring("\$\\beta_{ipsi}\$"))
#plt.text(-4, 12, "F")
plt.savefig("ipsi_strong_contra_weak_contra_d0_dms_split.pdf", bbox_inches="tight", transparent=true)


#############################################################################################
#############################################################################################
#############################################################################################
################
# LINE PLOT BIAS
################
plt.figure(figsize=(5, 4))
xt = 1:20
mt = mt
strg_color="tab:red"
d = nanmean(bias_contra_mult[:, mt][:, strong_dms] , 2)
plt.plot(xt, d, lw=3, color=strg_color)
v = nansem(bias_contra_mult[:, mt][:, strong_dms], 2)
plt.fill_between(xt, d .- v, d .+ v, alpha=0.4, color=strg_color, label="_nolegend_")

wk_color="tab:orange"
d = nanmean(bias_contra_mult[:, mt][:, weak_dms] , 2)
plt.plot(xt, d, lw=3, color=wk_color)
v = nansem(bias_contra_mult[:, mt][:, weak_dms] , 2)
plt.fill_between(xt, d .- v, d .+ v, alpha=0.4, color=wk_color, label="_nolegend_")


plt.xticks([1, 10, 20])
#plt.yticks([-1, 0, 1, 2])
# plt.ylim(-2. ,10)

plt.xlabel("Training day")
plt.ylabel(latexstring("\$\\beta_{bias}\$"))

plt.savefig("strong_bias_weak_bias_d0split_dms.pdf", bbox_inches="tight", transparent=true)

#############################################################################################
#############################################################################################
#############################################################################################
################
# LINE PLOT CHOICE HISTORY
################
plt.figure(figsize=(5, 4))
xt = 1:20

strg_color="tab:red"
d = nanmean(chist[xt, mt][:, strong_dms], 2)
plt.plot(xt, d, lw=3, color=strg_color)
v = nansem(chist[xt, mt][:, strong_dms], 2)
plt.fill_between(xt, d .- v, d .+ v, alpha=0.4, color=strg_color, label="_nolegend_")

wk_color="tab:orange"
d = nanmean(chist[xt, mt][:, weak_dms], 2)
plt.plot(xt, d, lw=3, color=wk_color)
v = nansem(chist[xt, mt][:, weak_dms], 2)
plt.fill_between(xt, d .- v, d .+ v, alpha=0.4, color=wk_color, label="_nolegend_")


#############################################################################################
#############################################################################################
#############################################################################################
##################
# STATS CONTRA MODEL
##################
strong_contra_df_dms = df_for_stats(contra_dms.beh[xt, mt][:, strong_dms], 
                        day0_contra_dms.cm[strong_dms], fips[mt][strong_dms], "contra")

weak_contra_df_dms = df_for_stats(contra_dms.beh[xt, mt][:, weak_dms], 
                        day0_contra_dms.cm[weak_dms], fips[mt][weak_dms], "contra")

contra_df_dms = [strong_contra_df_dms; weak_contra_df_dms]
contra_df_dms[!, :mc_neural_strength] = contra_df_dms.neural_strength .- mean(contra_df_dms.neural_strength);

frm = @formula(contrast_weight ~ day_split3*mc_neural_strength + (1 + day_split3|mouse))
contrasts = Dict(
    :mouse => DummyCoding(), 
    :day_split3 => DummyCoding()
)
contra_model = lme(frm, contra_df_dms, contrasts=contrasts)
anova(contra_model, type=3)
    

##################
# STATS IPSI MODEL
##################
strong_ipsi_df_dms = df_for_stats(ipsi_dms.beh[xt, mt][:, strong_dms], 
                        day0_contra_dms.cm[strong_dms], fips[mt][strong_dms], "contra")

weak_ipsi_df_dms = df_for_stats(ipsi_dms.beh[xt, mt][:, weak_dms], 
                        day0_contra_dms.cm[weak_dms], fips[mt][weak_dms], "contra")

ipsi_df_dms = [strong_ipsi_df_dms; weak_ipsi_df_dms]
ipsi_df_dms[!, :mc_neural_strength] = ipsi_df_dms.neural_strength .- mean(ipsi_df_dms.neural_strength);

frm = @formula(contrast_weight ~ day_split3*mc_neural_strength + (1 + day_split3|mouse))
contrasts = Dict(
    :mouse => DummyCoding(), 
    :day_split3 => DummyCoding()
)
ipsi_model = lme(frm, ipsi_df_dms, contrasts=contrasts)
anova(ipsi_model, type=3)

#############################################################################################
#############################################################################################
#############################################################################################
##################
# STATS bias model
##################
strong_bias_df_dms = df_for_stats_bias_chist(bias_contra_mult[xt, mt][:, strong_dms] , 
                        day0_contra_dms.cm[strong_dms], fips[mt][strong_dms], "contra")

weak_bias_df_dms = df_for_stats_bias_chist(bias_contra_mult[xt, mt][:, weak_dms] , 
                        day0_contra_dms.cm[weak_dms], fips[mt][weak_dms], "contra")

bias_df_dms = [strong_bias_df_dms; weak_bias_df_dms];

bias_df_dms[!, :mc_d0_strength] = bias_df_dms.d0_strength .- nanmean(bias_df_dms.d0_strength)

frm = @formula(dep_weight ~ day_split3*mc_d0_strength + (1 + day_split3|mouse))
contrasts = Dict(
    :mouse => DummyCoding(), 
    :day_split3 => DummyCoding()
)
bias_model = lme(frm, bias_df_dms, contrasts=contrasts)
anova(bias_model, type=3)

#############################################################################################
#############################################################################################
#############################################################################################
##################
# STATS choice history model
##################
strong_chist_df_dms = df_for_stats_bias_chist(chist[xt, mt][:, strong_dms], 
                        day0_contra_dms.cm[strong_dms], fips[mt][strong_dms], "contra")

weak_chist_df_dms = df_for_stats_bias_chist(chist[xt, mt][:, weak_dms], 
                        day0_contra_dms.cm[weak_dms], fips[mt][weak_dms], "contra")

chist_df_dms = [strong_chist_df_dms; weak_chist_df_dms];

chist_df_dms[!, :mc_d0_strength] = chist_df_dms.d0_strength .- nanmean(chist_df_dms.d0_strength)

frm = @formula(dep_weight ~ day_split3*mc_d0_strength + (1 + day_split3|mouse))
contrasts = Dict(
    :mouse => DummyCoding(), 
    :day_split3 => DummyCoding()
)
chist_model = lme(frm, chist_df_dms, contrasts=contrasts)
anova(chist_model, type=3)
