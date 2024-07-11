include("/Users/ysa/Documents/da_paper_code/fig/paper_analysis_prereq.jl");
include("/Users/ysa/Documents/da_paper_code/fig/figboilerplate.jl");

function rsquared(y, yhat)
    y_bar = mean(y)
    quo = sum((y .- yhat).^2) / sum((y .- y_bar).^2)
    return 1 - quo
end

cor_by_mouse = Dict(
    "dms" => Dict(
        "contra" => [robust_nanlm(contra_dms.cm[:, j], contra_dms.beh[:, j]).cor_api for j in 1:22],
        "ipsi" => [robust_nanlm(ipsi_dms.cm[:, j], ipsi_dms.beh[:, j]).cor_api for j in 1:22],
        "mismatched" => [robust_nanlm(contra_dms.cm[:, j], ipsi_dms.beh[:, j]).cor_api for j in 1:22]
    ),
    "dls" => Dict(
        "contra" => [robust_nanlm(contra_dls.cm[:, j], contra_dls.beh[:, j]).cor_api for j in 1:22],
        "ipsi" => [robust_nanlm(ipsi_dls.cm[:, j], ipsi_dls.beh[:, j]).cor_api for j in 1:22],
        "mismatched" => [robust_nanlm(contra_dls.cm[:, j], ipsi_dls.beh[:, j]).cor_api for j in 1:22]

    ),
    "nacc" => Dict(
        "contra" => [robust_nanlm(contra_nacc.cm[:, j], contra_nacc.beh[:, j]).cor_api for j in 1:22],
        "ipsi" => [robust_nanlm(ipsi_nacc.cm[:, j], ipsi_nacc.beh[:, j]).cor_api for j in 1:22],
        "mismatched" => [robust_nanlm(contra_nacc.cm[:, j], ipsi_nacc.beh[:, j]).cor_api for j in 1:22]

    )
    );

    dms_cor_df = pd.DataFrame(
    hcat([
        vcat([[cor_by_mouse["dms"]["contra"]; cor_by_mouse["dms"]["ipsi"]]]...),
        [["contra \n v.s. contra" for i in 1:n_mice]; ["ipsi \n v.s. ipsi" for i in 1:n_mice]],
        [fips; fips]
            
        ]...),

    columns=["correlation", "DMS", "mouse"]
    )

dls_cor_df = pd.DataFrame(
    hcat([
        vcat([[cor_by_mouse["dls"]["contra"]; cor_by_mouse["dls"]["ipsi"]]]...),
        [["contra \n v.s. contra" for i in 1:n_mice]; ["ipsi \n v.s. ipsi" for i in 1:n_mice]],
        [fips; fips]
            
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
            ax=ax, zorder=10)
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
plt.ylim(-1, 1.1)
plt.legend([], frameon=false)
ax.axhline(0, linestyle="--", lw=2, color="black")
ax.set_yticks([])
plt.xlabel("DLS")
#plt.ylabel("correlation")
plt.ylabel("")
plt.savefig("PAPER_FINAL_DLS_daybyday_cor_STIM_100-6.pdf", transparent=true, bbox_inches="tight")