include(joinpath(@__DIR__, "pipeline/base/utility.jl"))
using JLD2;

# load results
results = load("/Users/ysa/Desktop/pipeline/saved_results/neural_results.jld2", "results");
kernel_norm = results.kernel_norm;

mouse_ids = [collect(13:16); collect(26:43)];

n_days = 20
n_sim = 4
n_mice = 22
contra_stim_dms = [kernel_norm[mouse]["DMS"][dms_contra_map[mouse]] for mouse in mouse_ids];
contra_stim_dms = reshape(hcat(contra_stim_dms...), n_days, n_sim, n_mice);
avg_contra_dms = nanmean(contra_stim_dms, 3);
