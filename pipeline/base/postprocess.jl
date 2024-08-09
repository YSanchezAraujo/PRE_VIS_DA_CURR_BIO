include("utility.jl");
using JLD2;

# load results
results = load("neural_results.jld2", "results");
kernel_norm = results.kernel_norm;

mouse_ids = [collect(13:16); collect(26:43)];

contra_stim_dms = [results_Knorm[mouse]["DLS"][dms_contra_map[mouse]] for mouse in mouse_ids];
contra_stim_dms = reshape(hcat(contra_stim_dms...), 20, 4, 22);
avg_contra_dms = nanmean(contra_stim_dms, 3);
