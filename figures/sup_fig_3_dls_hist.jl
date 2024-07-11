include("/Users/ysa/paper_analysis_prereq.jl");

# HISTOGRAM
#day0contraDLS = day0norms["contra"]["dls"][:, 4] .- day0norms["contra"]["dls"][:, 1];
mask = .!isnan.(day0_contra_dls.cm)
kmfit = kmeans(day0_contra_dls.cm[mask]', 2)
cents = vec(kmfit.centers)
if cents[1] < cents[2]
    strong_dls = kmfit.assignments .== 2
    weak_dls = kmfit.assignments .== 1
else
    strong_dls = kmfit.assignments .== 1
    weak_dls = kmfit.assignments .== 2
end

strong_dls = day0contraDLS .> median(day0contraDLS)
weak_dls = .!strong_dls


plt.hist(day0_contra_dls.cm[mask][weak_dls], bins=4,
    color="tab:green", edgecolor="white", label="weak")

plt.hist(day0_contra_dls.cm[mask][strong_dls], bins=4,
    color="darkgreen", edgecolor="white", label="strong")

plt.axvline(0.57, linestyle="--", color="black", lw=3)
plt.xlabel(latexstring("\$ ||\\boldsymbol{K}^{100}_{stimulus} ||_2 -  || \\boldsymbol{K}^{6.25}_{stimulus} ||_2 \$"))
plt.ylabel("counts")
plt.title("Day 0 contra DLS")
plt.legend(["weak", "strong"], fontsize=18)
plt.savefig("day0_DLS_contra_hist.pdf", bbox_inches="tight", transparent=true)
#############################################################################################
#############################################################################################
#############################################################################################

# LINE PLOT CONTRA
xt = 1:20
mt = 5:22
plt.figure(figsize=(5, 4))
strg_color="darkgreen"
d = nanmean(contra_dls.beh[:, mt[mask]][:, strong_dls], 2)
plt.plot(xt, d, lw=3, color=strg_color)
v = nansem(contra_dls.beh[:, mt[mask]][:, strong_dls], 2) 
plt.fill_between(xt, d .- v, d .+ v, alpha=0.4, color=strg_color, label="_nolegend_")

wk_color="tab:green"
d = nanmean(contra_dls.beh[:, mt[mask]][:, weak_dls], 2)
plt.plot(xt, d, lw=3, color=wk_color)
v = nansem(contra_dls.beh[:, mt[mask]][:, weak_dls], 2)
plt.fill_between(xt, d .- v, d .+ v, alpha=0.4, color=wk_color, label="_nolegend_")


plt.xticks([1, 10, 20])
#plt.yticks([0, 5, 8])
# plt.ylim(-2. ,10)
#plt.legend(["strong", "weak"], fontsize=17, loc="upper left")
plt.xlabel("Session")
plt.ylabel(latexstring("\$\\beta_{contra}\$"))
#plt.text(-4, 12, "F")
plt.savefig("strong_contra_weak_contra_d0_DLS_split.pdf", bbox_inches="tight", transparent=true)
#############################################################################################
#############################################################################################
#############################################################################################

# LINE PLOT IPSI
plt.figure(figsize=(5,4))
strg_color="darkgreen"
d = nanmean(ipsi_dls.beh[:, mt[mask]][:, strong_dls], 2)
plt.plot(xt, d, lw=3, color=strg_color)
v = nansem(ipsi_dls.beh[:, mt[mask]][:, strong_dls], 2) 
plt.fill_between(xt, d .- v, d .+ v, alpha=0.4, color=strg_color, label="_nolegend_")

wk_color="tab:green"
d = nanmean(ipsi_dls.beh[:, mt[mask]][:, weak_dls], 2)
plt.plot(xt, d, lw=3, color=wk_color)
v = nansem(ipsi_dls.beh[:, mt[mask]][:, weak_dls], 2)
plt.fill_between(xt, d .- v, d .+ v, alpha=0.4, color=wk_color, label="_nolegend_")


plt.xticks([1, 10, 20])
# plt.ylim(-2. ,10)
#plt.legend(["strong", "weak"], fontsize=17, loc="upper left")
plt.xlabel("Session")
plt.ylabel(latexstring("\$\\beta_{ipsi}\$"))
#plt.text(-4, 12, "F")
plt.savefig("ipsi_strong_contra_weak_contra_d0_dls_split.pdf", bbox_inches="tight", transparent=true)


#############################################################################################
#############################################################################################
#############################################################################################
################
# LINE PLOT BIAS
################
plt.figure(figsize=(5, 4))
xt = 1:20
mt = mt
strg_color="darkgreen"
d = nanmean(bias_contra_mult[:, mt[mask]][:, strong_dls] * -1, 2)
plt.plot(xt, d, lw=3, color=strg_color)
v = nansem(bias_contra_mult[:, mt[mask]][:, strong_dls] * -1, 2)
plt.fill_between(xt, d .- v, d .+ v, alpha=0.4, color=strg_color, label="_nolegend_")

wk_color="tab:green"
d = nanmean(bias_contra_mult[:, mt[mask]][:, weak_dls] * -1, 2)
plt.plot(xt, d, lw=3, color=wk_color)
v = nansem(bias_contra_mult[:, mt[mask]][:, weak_dls] * -1, 2)
plt.fill_between(xt, d .- v, d .+ v, alpha=0.4, color=wk_color, label="_nolegend_")


plt.xticks([1, 10, 20])
#plt.yticks([-1, 0, 1, 2])
# plt.ylim(-2. ,10)

plt.xlabel("Session")
plt.ylabel(latexstring("\$\\beta_{bias}\$"))

plt.savefig("strong_bias_weak_bias_d0split_dls.pdf", bbox_inches="tight", transparent=true)

#############################################################################################
#############################################################################################
#############################################################################################
################
# LINE PLOT CHOICE HISTORY
################
plt.figure(figsize=(5, 4))
xt = 1:20

strg_color="darkgreen"
d = nanmean(chist[xt, mt[mask]][:, strong_dls], 2)
plt.plot(xt, d, lw=3, color=strg_color)
v = nansem(chist[xt, mt[mask]][:, strong_dls], 2)
plt.fill_between(xt, d .- v, d .+ v, alpha=0.4, color=strg_color, label="_nolegend_")

wk_color="lightgreen"
d = nanmean(chist[xt, mt[mask]][:, weak_dls], 2)
plt.plot(xt, d, lw=3, color=wk_color)
v = nansem(chist[xt, mt[mask]][:, weak_dls], 2)
plt.fill_between(xt, d .- v, d .+ v, alpha=0.4, color=wk_color, label="_nolegend_")

plt.xlabel("Session")
plt.ylabel(latexstring("\$\\beta_{choice history}\$"))

plt.savefig("strong_CHIST_weak_CHIST_d0split_dls.pdf", bbox_inches="tight", transparent=true)

#############################################################################################
#############################################################################################
#############################################################################################
##################
# STATS CONTRA MODEL
##################
strong_contra_df_dls = df_for_stats(contra_dls.beh[xt, mt[mask]][:, strong_dls], 
                        day0_contra_dls.cm[mask][strong_dls], fips[mt[mask]][strong_dls], "contra")

weak_contra_df_dls = df_for_stats(contra_dls.beh[xt, mt[mask]][:, weak_dls], 
                        day0_contra_dls.cm[mask][weak_dls], fips[mt[mask]][weak_dls], "contra")

contra_df_dls = [strong_contra_df_dls; weak_contra_df_dls]
contra_df_dls[!, :mc_neural_strength] = contra_df_dls.neural_strength .- mean(contra_df_dls.neural_strength);

frm = @formula(contrast_weight ~ day_split3*mc_neural_strength + (1 + day_split3|mouse))
contrasts = Dict(
    :mouse => DummyCoding(), 
    :day_split3 => DummyCoding()
)
contra_model = lme(frm, contra_df_dls, contrasts=contrasts)
anova(contra_model, type=3)
    

##################
# STATS IPSI MODEL
##################
strong_ipsi_df_dls = df_for_stats(ipsi_dls.beh[xt, mt[mask]][:, strong_dls], 
                        day0_contra_dls.cm[mask][strong_dls], fips[mt[mask]][strong_dls], "contra")

weak_ipsi_df_dls = df_for_stats(ipsi_dls.beh[xt, mt[mask]][:, weak_dls], 
                        day0_contra_dls.cm[mask][weak_dls], fips[mt[mask]][weak_dls], "contra")

ipsi_df_dls = [strong_ipsi_df_dls; weak_ipsi_df_dls]
ipsi_df_dls[!, :mc_neural_strength] = ipsi_df_dls.neural_strength .- mean(ipsi_df_dls.neural_strength);

frm = @formula(contrast_weight ~ day_split3*mc_neural_strength + (1 + day_split3|mouse))
contrasts = Dict(
    :mouse => DummyCoding(), 
    :day_split3 => DummyCoding()
)
ipsi_model = lme(frm, ipsi_df_dls, contrasts=contrasts)
anova(ipsi_model, type=3)

#############################################################################################
#############################################################################################
#############################################################################################
##################
# STATS bias model
##################
strong_bias_df_dls = df_for_stats_bias_chist(bias_contra_mult[xt, mt[mask]][:, strong_dls] * -1, 
                        day0_contra_dls.cm[mask][strong_dls], fips[mt[mask]][strong_dls], "contra")

weak_bias_df_dls = df_for_stats_bias_chist(bias_contra_mult[xt, mt[mask]][:, weak_dls] * -1, 
                        day0_contra_dls.cm[mask][weak_dls], fips[mt[mask]][weak_dls], "contra")

bias_df_dls = [strong_bias_df_dls; weak_bias_df_dls];

bias_df_dls[!, :mc_d0_strength] = bias_df_dls.d0_strength .- nanmean(bias_df_dls.d0_strength)

frm = @formula(dep_weight ~ day_split3*mc_d0_strength + (1 + day_split3|mouse))
contrasts = Dict(
    :mouse => DummyCoding(), 
    :day_split3 => DummyCoding()
)
bias_model = lme(frm, bias_df_dls, contrasts=contrasts)
anova(bias_model, type=3)

#############################################################################################
#############################################################################################
#############################################################################################
##################
# STATS choice history model
##################
strong_chist_df_dls = df_for_stats_bias_chist(chist[xt, mt[mask]][:, strong_dls], 
                        day0_contra_dls.cm[mask][strong_dls], fips[mt[mask]][strong_dls], "contra")

weak_chist_df_dls = df_for_stats_bias_chist(chist[xt, mt[mask]][:, weak_dls], 
                        day0_contra_dls.cm[mask][weak_dls], fips[mt[mask]][weak_dls], "contra")

chist_df_dls = [strong_chist_df_dls; weak_chist_df_dls];

chist_df_dls[!, :mc_d0_strength] = chist_df_dls.d0_strength .- nanmean(chist_df_dls.d0_strength)

frm = @formula(dep_weight ~ day_split3*mc_d0_strength + (1 + day_split3|mouse))
contrasts = Dict(
    :mouse => DummyCoding(), 
    :day_split3 => DummyCoding()
)
chist_model = lme(frm, chist_df_dls, contrasts=contrasts)
anova(chist_model, type=3)
