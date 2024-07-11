type_path = "mac";
fbp_path = Dict( 
    "clust" => "/jukebox/witten/yoel/julia/analysis_files/dummy_testing/figboilerplate.jl",
    "mac" => "/Users/ysa/Documents/da_paper_code/fig/figboilerplate.jl"
)[type_path];

include(fbp_path);

choice_path = Dict(
    "clust" => "/jukebox/witten/yoel/julia/analysis_files/dummy_testing/choice_res_cohalpha_nu_20.jld2",
    "mac" => "/Users/ysa/Documents/da_paper_code/data/cw_matched_day.jld2"
)[type_path];

choice_weights = load(choice_path, "choice_weights");
lrs  = load("/Users/ysa/Documents/da_paper_code/data/lrs.jld2", "lr");
cohalpha = load("/Users/ysa/Documents/da_paper_code/data/cohalpha.jld2", "ca")
daymap = load("/Users/ysa/Documents/da_paper_code/data/day_map.jld2", "map")



neu_path = Dict(
    "clust" => "/jukebox/witten/yoel/julia/analysis_files/dummy_testing/neu_full_res_only.jld2",
    "mac" => "/Users/ysa/Documents/da_paper_code/data/neu_weights_matched.jld2"
)[type_path];

neu = load(neu_path, "weights");

pretrain_path = Dict(
    "clust" => "/jukebox/witten/yoel/julia/analysis_files/dummy_testing/pretrain.jld2",
    "mac" => "/Users/ysa/Documents/da_paper_code/data/pretrain_trial_data.jld2"
)[type_path];

pretrain = load(pretrain_path, "data");

pfips = collect(26:43);

cstat_path = Dict(
    "clust" => "/jukebox/witten/yoel/julia/analysis_files/dummy_testing/cstat_df.jld2",
    "mac" => "/Users/ysa/Documents/da_paper_code/data/cstat_df.jld2"
)[type_path];

cstat = load(cstat_path, "cstat");

day0norms = load("/Users/ysa/Documents/da_paper_code/data/d0_norms_ci.jld2", "d0norms_ci");
day0trialdata = load("/Users/ysa/Documents/da_paper_code/data/avg_trial_data.jld2", "avg_trial_data");
day0mats = (
    nacc20 = load("/Users/ysa/Documents/da_paper_code/data/d0avg_mat_nacc.jld2", "nacc"),
    dls20 = load("/Users/ysa/Documents/da_paper_code/data/d0avg_mat_dls.jld2", "dls"),
    nacc25 = load("/Users/ysa/Documents/da_paper_code/data/d0avg_mat_nacc.jld2", "nacc"),
    dls25 = load("/Users/ysa/Documents/da_paper_code/data/d0avg_mat_dls.jld2", "dls")
)