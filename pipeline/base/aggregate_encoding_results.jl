include(joinpath(@__DIR__, "preprocess.jl"))
include(joinpath(@__DIR__, "constants.jl"))

save_path = "/jukebox/witten/yoel/saved_results"

load_func(save_path, mouse) = load(
    joinpath(save_path, "neural_results_mouseid_$(mouse).jld2"),
     "results"
)

mouse_ids = [collect(13:16); collect(26:43)];

results_K = Dict();
results_E = Dict();
results_Knorm = Dict();
results_Enorm = Dict();
results_vexpl = Dict();

for mouse in mouse_ids
    mouse_result = load_func(save_path, mouse);
    setindex!(results_K, mouse_result.kernel, mouse)
    setindex!(results_E, mouse_result.error, mouse)
    setindex!(results_Knorm, mouse_result.kernel_norm, mouse)
    setindex!(results_Enorm, mouse_result.error_norm, mouse)
    setindex!(results_vexpl, mouse_result.vexpl, mouse)
end

using JLD2;
save(
    joinpath(save_path, "neural_results.jld2"),
     "results",
    (
        kernel = results_K,
        error = results_E,
        kernel_norm = results_Knorm,
        error_norm = results_Enorm,
        vexpl = results_vexpl

    )
)
