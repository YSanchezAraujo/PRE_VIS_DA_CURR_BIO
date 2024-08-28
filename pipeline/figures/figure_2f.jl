include(joinpath(@__DIR__, "../base/preprocess.jl"));
include(joinpath(@__DIR__, "../base/constants.jl"));
include(joinpath(@__DIR__, "../base/correlation_functions.jl"));
include(joinpath(@__DIR__, "../base/utility.jl"));
include(joinpath(@__DIR__, "../base/figboilerplate.jl"));

using HypothesisTests;

neu_results = load("/jukebox/witten/yoel/saved_results/neural_results.jld2", "results");
behavior_weight = load("/jukebox/witten/yoel/saved_results/choice_weights.jld2", "results");
kernel_norm = neu_results.kernel_norm;
mouse_ids = [collect(13:16); collect(26:43)];
n_mice = length(mouse_ids);

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

contra_dms_corrs = avg_contra_correlation_conmod_dms(kernel_norm, behavior_weight, mouse_ids)
contra_dls_corrs = avg_contra_correlation_conmod_dls(kernel_norm, behavior_weight, mouse_ids)
contra_nacc_corrs = avg_contra_correlation_conmod_nacc(kernel_norm, behavior_weight, mouse_ids)

ipsi_dms_corrs = avg_ipsi_correlation_conmod_dms(kernel_norm, behavior_weight, mouse_ids)
ipsi_dls_corrs = avg_ipsi_correlation_conmod_dls(kernel_norm, behavior_weight, mouse_ids)
ipsi_nacc_corrs = avg_ipsi_correlation_conmod_nacc(kernel_norm, behavior_weight, mouse_ids)


# stats within and between groups

# DMS null hypothesis of no difference from zero
null_zero_dms_contra = OneSampleTTest(contra_dms_corrs.corrs)
null_zero_dms_ipsi = OneSampleTTest(ipsi_dms_corrs.corrs)

# DLS null hypothesis of no difference from zero
null_zero_dls_contra = OneSampleTTest(contra_dls_corrs.corrs)
null_zero_dls_ipsi = OneSampleTTest(ipsi_dls_corrs.corrs)

# NAcc null hypothesis of no difference from zero
null_zero_nacc_contra = OneSampleTTest(contra_nacc_corrs.corrs)
null_zero_nacc_ipsi = OneSampleTTest(ipsi_nacc_corrs.corrs)


diff_dms = OneSampleTTest(contra_dms_corrs.corrs .- ipsi_dms_corrs.corrs)
diff_dls = OneSampleTTest(contra_dls_corrs.corrs .- ipsi_dls_corrs.corrs)
diff_nacc = OneSampleTTest(contra_nacc_corrs.corrs .- ipsi_nacc_corrs.corrs)


# DMS v.s. DLS null hypothesis of no difference of differences between regions
diff_of_diff_dms_dls = OneSampleTTest(
    (contra_dms_corrs.corrs .- ipsi_dms_corrs.corrs) .- (contra_dls_corrs.corrs .- ipsi_dls_corrs.corrs)
)

# DMS v.s. NAcc null hypothesis of no difference of differences between regions
diff_of_diff_dms_nacc = OneSampleTTest(
    (contra_dms_corrs.corrs .- ipsi_dms_corrs.corrs) .- (contra_nacc_corrs.corrs .- ipsi_nacc_corrs.corrs)
)


pd = pyimport("pandas");

cor_by_mouse = Dict(
    "dms" => Dict(
        "contra" => contra_dms_corrs.corrs,
        "ipsi" => ipsi_dms_corrs.corrs,
    ),
    "dls" => Dict(
        "contra" => contra_dls_corrs.corrs,
        "ipsi" => ipsi_dls_corrs.corrs,

    ),
    "nacc" => Dict(
        "contra" => contra_nacc_corrs.corrs,
        "ipsi" => ipsi_nacc_corrs.corrs,
    )
);

# DMS subplot
dms_cor_df = pd.DataFrame(
    hcat([
        vcat([[cor_by_mouse["dms"]["contra"]; cor_by_mouse["dms"]["ipsi"]]]...),
        [["Contra \n v.s. Contra" for i in 1:n_mice]; ["Ipsi \n v.s. Ipsi" for i in 1:n_mice]],
        [mouse_ids; mouse_ids]
        ]...),

        columns=["correlation", "DMS", "mouse"]
)

plt.figure(figsize=(3, 5))
ax=sns.swarmplot(data=dms_cor_df, x="DMS", y="correlation", color="tab:orange", s=8,linewidth=1, zorder=1)

sns.boxplot(
            x="DMS",
            y="correlation",
            data=dms_cor_df,
            showmeans=true,
            meanline=true,
            meanprops=Dict("color" => "k", "ls" => "-", "lw" => 2),
            medianprops=Dict("visible" => false),
            whiskerprops=Dict("linewidth" => 0, "zorder" =>10),
            boxprops=Dict("facecolor" => nothing, "zorder" => 10, "fill" => nothing, "lw" => 3),
            showfliers=false,
            showcaps=false,
            ax=ax, zorder=10
)

ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
plt.ylim(-1, 1.1)
plt.legend([], frameon=false)
ax.axhline(0, linestyle="--", lw=2, color="black")
ax.set_yticks([])
plt.xlabel("DMS")
plt.ylabel("")
plt.savefig("figure_2f_DMS_daybyday_cor_STIM_100-6.pdf", transparent=true, bbox_inches="tight")
plt.close()

# DLS subplot
dls_cor_df = pd.DataFrame(
    hcat([
        vcat([[cor_by_mouse["dls"]["contra"]; cor_by_mouse["dls"]["ipsi"]]]...),
        [["Contra \n v.s. Contra" for i in 1:n_mice]; ["Ipsi \n v.s. Ipsi" for i in 1:n_mice]],
        [mouse_ids; mouse_ids]
        ]...),

        columns=["correlation", "DLS", "mouse"]
)

plt.figure(figsize=(3, 5))
ax=sns.swarmplot(data=dls_cor_df, x="DLS", y="correlation", color="tab:green", s=8,linewidth=1, zorder=1)

sns.boxplot(
            x="DLS",
            y="correlation",
            data=dls_cor_df,
            showmeans=true,
            meanline=true,
            meanprops=Dict("color" => "k", "ls" => "-", "lw" => 2),
            medianprops=Dict("visible" => false),
            whiskerprops=Dict("linewidth" => 0, "zorder" =>10),
            boxprops=Dict("facecolor" => nothing, "zorder" => 10, "fill" => nothing, "lw" => 3),
            showfliers=false,
            showcaps=false,
            ax=ax, zorder=10
)

ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
plt.ylim(-1, 1.1)
plt.legend([], frameon=false)
ax.axhline(0, linestyle="--", lw=2, color="black")
ax.set_yticks([])
plt.xlabel("DLS")
plt.ylabel("")
plt.savefig("figure_2f_DLS_daybyday_cor_STIM_100-6.pdf", transparent=true, bbox_inches="tight")
plt.close()
# NAcc subplot
nacc_cor_df = pd.DataFrame(
    hcat([
        vcat([[cor_by_mouse["nacc"]["contra"]; cor_by_mouse["nacc"]["ipsi"]]]...),
        [["Contra \n v.s. Contra" for i in 1:n_mice]; ["Ipsi \n v.s. Ipsi" for i in 1:n_mice]],
        [mouse_ids; mouse_ids]
        ]...),

        columns=["correlation", "NAcc", "mouse"]
)

plt.figure(figsize=(3, 5))
ax=sns.swarmplot(data=nacc_cor_df, x="NAcc", y="correlation", color="tab:blue", s=8,linewidth=1, zorder=1)

sns.boxplot(
            x="NAcc",
            y="correlation",
            data=nacc_cor_df,
            showmeans=true,
            meanline=true,
            meanprops=Dict("color" => "k", "ls" => "-", "lw" => 2),
            medianprops=Dict("visible" => false),
            whiskerprops=Dict("linewidth" => 0, "zorder" =>10),
            boxprops=Dict("facecolor" => nothing, "zorder" => 10, "fill" => nothing, "lw" => 3),
            showfliers=false,
            showcaps=false,
            ax=ax, zorder=10
)

ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
plt.ylim(-1, 1.1)
plt.legend([], frameon=false)
ax.axhline(0, linestyle="--", lw=2, color="black")
plt.xlabel("NAcc")
plt.ylabel("")
plt.savefig("figure_2f_NAcc_daybyday_cor_STIM_100-6.pdf", transparent=true, bbox_inches="tight")
plt.close()