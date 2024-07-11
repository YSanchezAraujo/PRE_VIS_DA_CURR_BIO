include("/Users/ysa/paper_analysis_prereq.jl");

# HISTOGRAM
#day0contranacc = day0norms["contra"]["nacc"][:, 4] .- day0norms["contra"]["nacc"][:, 1];
mask = .!isnan.(day0_contra_nacc.cm)
kmfit = kmeans(day0_contra_nacc.cm[mask]', 2)

cents = vec(kmfit.centers)
if cents[1] < cents[2]
    strong_nacc = kmfit.assignments .== 2
    weak_nacc = kmfit.assignments .== 1
else
    strong_nacc = kmfit.assignments .== 1
    weak_nacc = kmfit.assignments .== 2
end

#strong_nacc = day0contranacc .> median(day0contranacc)
#weak_nacc = .!strong_nacc


plt.hist(day0_contra_nacc.cm[mask][weak_nacc], bins=4,
    color="lightblue", edgecolor="white", label="weak")

plt.hist(day0_contra_nacc.cm[mask][strong_nacc], bins=4,
    color="tab:blue", edgecolor="white", label="strong")

plt.axvline(0.2, linestyle="--", color="black", lw=3)
plt.xlabel(latexstring("\$ ||\\boldsymbol{K}^{100}_{stimulus} ||_2 -  || \\boldsymbol{K}^{6.25}_{stimulus} ||_2 \$"))
plt.ylabel("counts")
plt.title("Day 0 contra NAc")
plt.legend(["weak", "strong"], fontsize=18)
plt.savefig("day0_nacc_contra_hist.pdf", bbox_inches="tight", transparent=true)
#############################################################################################
#############################################################################################
#############################################################################################

# LINE PLOT CONTRA
xt = 1:20
mt = 5:22
plt.figure(figsize=(5, 4))
strg_color="tab:blue"
d = nanmean(contra_nacc.beh[:, mt[mask]][:, strong_nacc], 2)
plt.plot(xt, d, lw=3, color=strg_color)
v = nansem(contra_nacc.beh[:, mt[mask]][:, strong_nacc], 2) 
plt.fill_between(xt, d .- v, d .+ v, alpha=0.4, color=strg_color, label="_nolegend_")

wk_color="lightblue"
d = nanmean(contra_nacc.beh[:, mt[mask]][:, weak_nacc], 2)
plt.plot(xt, d, lw=3, color=wk_color)
v = nansem(contra_nacc.beh[:, mt[mask]][:, weak_nacc], 2)
plt.fill_between(xt, d .- v, d .+ v, alpha=0.4, color=wk_color, label="_nolegend_")


plt.xticks([1, 10, 20])
#plt.yticks([0, 5, 8])
# plt.ylim(-2. ,10)
#plt.legend(["strong", "weak"], fontsize=17, loc="upper left")
plt.xlabel("Training day")
plt.ylabel(latexstring("\$\\beta_{contra}\$"))
#plt.text(-4, 12, "F")
plt.savefig("strong_contra_weak_contra_d0_nacc_split.pdf", bbox_inches="tight", transparent=true)
#############################################################################################
#############################################################################################
#############################################################################################

# LINE PLOT IPSI
plt.figure(figsize=(5,4))
strg_color="tab:blue"
d = nanmean(ipsi_nacc.beh[:, mt[mask]][:, strong_nacc], 2)
plt.plot(xt, d, lw=3, color=strg_color)
v = nansem(ipsi_nacc.beh[:, mt[mask]][:, strong_nacc], 2) 
plt.fill_between(xt, d .- v, d .+ v, alpha=0.4, color=strg_color, label="_nolegend_")

wk_color="lightblue"
d = nanmean(ipsi_nacc.beh[:, mt[mask]][:, weak_nacc], 2)
plt.plot(xt, d, lw=3, color=wk_color)
v = nansem(ipsi_nacc.beh[:, mt[mask]][:, weak_nacc], 2)
plt.fill_between(xt, d .- v, d .+ v, alpha=0.4, color=wk_color, label="_nolegend_")


plt.xticks([1, 10, 20])
# plt.ylim(-2. ,10)
#plt.legend(["strong", "weak"], fontsize=17, loc="upper left")
plt.xlabel("Training day")
plt.ylabel(latexstring("\$\\beta_{ipsi}\$"))
#plt.text(-4, 12, "F")
plt.savefig("ipsi_strong_contra_weak_contra_d0_nacc_split.pdf", bbox_inches="tight", transparent=true)

#############################################################################################
#############################################################################################
#############################################################################################
################
# LINE PLOT BIAS
################
plt.figure(figsize=(5, 4))
xt = 1:20
mt = mt
strg_color="tab:blue"
d = nanmean(bias_contra_mult[:, mt[mask]][:, strong_nacc] * -1, 2)
plt.plot(xt, d, lw=3, color=strg_color)
v = nansem(bias_contra_mult[:, mt[mask]][:, strong_nacc] * -1, 2)
plt.fill_between(xt, d .- v, d .+ v, alpha=0.4, color=strg_color, label="_nolegend_")

wk_color="lightblue"
d = nanmean(bias_contra_mult[:, mt[mask]][:, weak_nacc] * -1, 2)
plt.plot(xt, d, lw=3, color=wk_color)
v = nansem(bias_contra_mult[:, mt[mask]][:, weak_nacc] * -1, 2)
plt.fill_between(xt, d .- v, d .+ v, alpha=0.4, color=wk_color, label="_nolegend_")


plt.xticks([1, 10, 20])
#plt.yticks([-1, 0, 1, 2])
# plt.ylim(-2. ,10)

plt.xlabel("Training day")
plt.ylabel(latexstring("\$\\beta_{bias}\$"))

plt.savefig("strong_bias_weak_bias_d0split_nacc.pdf", bbox_inches="tight", transparent=true)

#############################################################################################
#############################################################################################
#############################################################################################
################
# LINE PLOT CHOICE HISTORY
################
plt.figure(figsize=(5, 4))
xt = 1:20

strg_color="tab:blue"
d = nanmean(chist[xt, mt[mask]][:, strong_nacc], 2)
plt.plot(xt, d, lw=3, color=strg_color)
v = nansem(chist[xt, mt[mask]][:, strong_nacc], 2)
plt.fill_between(xt, d .- v, d .+ v, alpha=0.4, color=strg_color, label="_nolegend_")

wk_color="lightblue"
d = nanmean(chist[xt, mt[mask]][:, weak_nacc], 2)
plt.plot(xt, d, lw=3, color=wk_color)
v = nansem(chist[xt, mt[mask]][:, weak_nacc], 2)
plt.fill_between(xt, d .- v, d .+ v, alpha=0.4, color=wk_color, label="_nolegend_")



#############################################################################################
#############################################################################################
#############################################################################################
##################
# STATS CONTRA MODEL
##################
strong_contra_df_nacc = df_for_stats(contra_nacc.beh[xt, mt[mask]][:, strong_nacc], 
                        day0_contra_nacc.cm[mask][strong_nacc], fips[mt[mask]][strong_nacc], "contra")

weak_contra_df_nacc = df_for_stats(contra_nacc.beh[xt, mt[mask]][:, weak_nacc], 
                        day0_contra_nacc.cm[mask][weak_nacc], fips[mt[mask]][weak_nacc], "contra")

contra_df_nacc = [strong_contra_df_nacc; weak_contra_df_nacc]
contra_df_nacc[!, :mc_neural_strength] = contra_df_nacc.neural_strength .- mean(contra_df_nacc.neural_strength);

frm = @formula(contrast_weight ~ day_split3*mc_neural_strength + (1 + day_split3|mouse))
contrasts = Dict(
    :mouse => DummyCoding(), 
    :day_split3 => DummyCoding()
)
contra_model = lme(frm, contra_df_nacc, contrasts=contrasts)
anova(contra_model, type=3)
    

##################
# STATS IPSI MODEL
##################
strong_ipsi_df_nacc = df_for_stats(ipsi_nacc.beh[xt, mt[mask]][:, strong_nacc], 
                        day0_contra_nacc.cm[mask][strong_nacc], fips[mt[mask]][strong_nacc], "contra")

weak_ipsi_df_nacc = df_for_stats(ipsi_nacc.beh[xt, mt[mask]][:, weak_nacc], 
                        day0_contra_nacc.cm[mask][weak_nacc], fips[mt[mask]][weak_nacc], "contra")

ipsi_df_nacc = [strong_ipsi_df_nacc; weak_ipsi_df_nacc]
ipsi_df_nacc[!, :mc_neural_strength] = ipsi_df_nacc.neural_strength .- mean(ipsi_df_nacc.neural_strength);

frm = @formula(contrast_weight ~ day_split3*mc_neural_strength + (1 + day_split3|mouse))
contrasts = Dict(
    :mouse => DummyCoding(), 
    :day_split3 => DummyCoding()
)
ipsi_model = lme(frm, ipsi_df_nacc, contrasts=contrasts)
anova(ipsi_model, type=3)

#############################################################################################
#############################################################################################
#############################################################################################
##################
# STATS bias model
##################
strong_bias_df_nacc = df_for_stats_bias_chist(bias_contra_mult[xt, mt[mask]][:, strong_nacc] * -1, 
                        day0_contra_nacc.cm[mask][strong_nacc], fips[mt[mask]][strong_nacc], "contra")

weak_bias_df_nacc = df_for_stats_bias_chist(bias_contra_mult[xt, mt[mask]][:, weak_nacc] * -1, 
                        day0_contra_nacc.cm[mask][weak_nacc], fips[mt[mask]][weak_nacc], "contra")

bias_df_nacc = [strong_bias_df_nacc; weak_bias_df_nacc];

bias_df_nacc[!, :mc_d0_strength] = bias_df_nacc.d0_strength .- nanmean(bias_df_nacc.d0_strength)

frm = @formula(dep_weight ~ day_split3*mc_d0_strength + (1 + day_split3|mouse))
contrasts = Dict(
    :mouse => DummyCoding(), 
    :day_split3 => DummyCoding()
)
bias_model = lme(frm, bias_df_nacc, contrasts=contrasts)
anova(bias_model, type=3)

#############################################################################################
#############################################################################################
#############################################################################################
##################
# STATS choice history model
##################
strong_chist_df_nacc = df_for_stats_bias_chist(chist[xt, mt[mask]][:, strong_nacc], 
                        day0_contra_nacc.cm[mask][strong_nacc], fips[mt[mask]][strong_nacc], "contra")

weak_chist_df_nacc = df_for_stats_bias_chist(chist[xt, mt[mask]][:, weak_nacc], 
                        day0_contra_nacc.cm[mask][weak_nacc], fips[mt[mask]][weak_nacc], "contra")

chist_df_nacc = [strong_chist_df_nacc; weak_chist_df_nacc];

chist_df_nacc[!, :mc_d0_strength] = chist_df_nacc.d0_strength .- nanmean(chist_df_nacc.d0_strength)

frm = @formula(dep_weight ~ day_split3*mc_d0_strength + (1 + day_split3|mouse))
contrasts = Dict(
    :mouse => DummyCoding(), 
    :day_split3 => DummyCoding()
)
chist_model = lme(frm, chist_df_nacc, contrasts=contrasts)
anova(chist_model, type=3)
