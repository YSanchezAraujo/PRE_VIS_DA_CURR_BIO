include(joinpath(@__DIR__, "encoding_model.jl"))
include(joinpath(@__DIR__, "preprocess.jl"))
include(joinpath(@__DIR__, "design_matrix.jl"))
include(joinpath(@__DIR__, "constants.jl"))


function scalar_summary_stats(weights, event_names, func)
    summary = Dict()

    for ev in event_names
        setindex!(summary, mapslices(func, weights[ev]; dims=1), ev)
    end

    return summary
end

function fit_mouse_day0(mouse_id, param_set, event_names, summary_func)
    nfunc, window, n_stim, n_sets, base_path = (
        param_set.nfunc, param_set.window, param_set.n_stim,
        param_set.n_sets, param_set.base_path
    )
    
    kernel_estimates = Dict(
        "NAcc" => Dict(ev => fill(NaN, window, n_stim) for ev in event_names[1:n_sets]), 
        "DMS" => Dict(ev => fill(NaN, window, n_stim) for ev in event_names[1:n_sets]),
        "DLS" => Dict(ev => fill(NaN, window, n_stim) for ev in event_names[1:n_sets])
    );

    error_estimates = Dict(
        "NAcc" => Dict(ev => fill(NaN, window, n_stim) for ev in event_names[1:n_sets]), 
        "DMS" => Dict(ev => fill(NaN, window, n_stim) for ev in event_names[1:n_sets]),
        "DLS" => Dict(ev => fill(NaN, window, n_stim) for ev in event_names[1:n_sets])
    );

    kernel_estimates_summaries = Dict(
        "NAcc" => Dict(ev => fill(NaN, n_stim) for ev in event_names[1:n_sets]), 
        "DMS" => Dict(ev => fill(NaN, n_stim) for ev in event_names[1:n_sets]),
        "DLS" => Dict(ev => fill(NaN, n_stim) for ev in event_names[1:n_sets])
    );

    error_estimates_summaries = Dict(
        "NAcc" => Dict(ev => fill(NaN, n_stim) for ev in event_names[1:n_sets]), 
        "DMS" => Dict(ev => fill(NaN, n_stim) for ev in event_names[1:n_sets]),
        "DLS" => Dict(ev => fill(NaN, n_stim) for ev in event_names[1:n_sets])
    );

    var_expl = Dict(
        "NAcc" => NaN, 
        "DMS" => NaN,
        "DLS" => NaN
    )

    data = mouse_session0_data(base_path, mouse_id);
    desmat_items = make_design_matrix_items_day0(data, window, nfunc)
    full_desmat = design_matrix_day0(desmat_items);

    # this design matrix is truncated to only behavioral events 
    trunc_desmat_items = truncate_design_matrix_day0(full_desmat, data, window);
    X = vcat(trunc_desmat_items.X...);
    Y = vcat(trunc_desmat_items.Y...);

    reg_labels = ["NAcc", "DMS", "DLS"]
    for (reg, rlabel) in enumerate(reg_labels)
        model_fit = bayes_ridge(X, Y[:, reg])
        # extract the weights in the standard basis
        W = desmat_items.basis * reshape(model_fit.w[2:end], (nfunc, n_stim * n_sets))
        S = desmat_items.basis * reshape(sqrt.(diag(model_fit.covar)[2:end]), (nfunc, n_stim * n_sets))

        kernels = weights_by_event(W, n_stim, event_names[1:n_sets])
        errors = weights_by_event(S, n_stim, event_names[1:n_sets])

        #compute summary statistics of the kernels
        kernel_norms = scalar_summary_stats(kernels, event_names[1:n_sets], summary_func)
        error_norms = scalar_summary_stats(errors, event_names[1:n_sets], summary_func)

        # compute rsquared
        var_expl[rlabel] = rsquared(Y[:, reg], X * model_fit.w)
        
        # store results
        for ev in event_names[1:n_sets]
            kernel_estimates[rlabel][ev] = kernels[ev]
            error_estimates[rlabel][ev] = errors[ev]

            kernel_estimates_summaries[rlabel][ev] = vec(kernel_norms[ev])
            error_estimates_summaries[rlabel][ev] = vec(error_norms[ev])
        end
    end
    

    return (
        kernel_estimates, 
        error_estimates, 
        kernel_estimates_summaries, 
        error_estimates_summaries,
        var_expl
    )
end

# set parameters
param_set = (
    nfunc = 16, # number of basis functions
    window = 50, # number of time bins for temporal kernels, spans 1 second
    n_stim = 4, # 4 stimulus levels: 6.25%, 12.5%, 25%, 100%
    n_sets = 2, # this corresponds to the number of events in the design matrix, see event names below
    base_path = "/jukebox/witten/ONE/alyx.internationalbrainlab.org/wittenlab/Subjects" # whereever you have downloaded the data
);

results_K = Dict();
results_E = Dict();
results_Knorm = Dict();
results_Enorm = Dict();
results_vexpl = Dict();

for mouse in collect(26:43)

    K, E, Knorm, Enorm, vexpl = fit_mouse_day0(mouse, param_set, event_names, norm);

    save_path = "/jukebox/witten/yoel/saved_results"
    if !isdir(save_path)
        mkpath(save_path)
        println("Directory created at: $save_path")
    else
        println("Directory already exists at: $save_path")
    end

    using JLD2;
    save(
        joinpath(save_path, "day0_neural_results_mouseid_$(mouse).jld2"),
        "results",
        (
            kernel = K,
            error = E,
            kernel_norm = Knorm,
            error_norm = Enorm,
            vexpl = vexpl
        )
    )

    setindex!(results_K, K, mouse)
    setindex!(results_E, E, mouse)
    setindex!(results_Knorm, Knorm, mouse)
    setindex!(results_Enorm, Enorm, mouse)
    setindex!(results_vexpl, vexpl, mouse)
end

save_path = "/jukebox/witten/yoel/saved_results"
save(
    joinpath(save_path, "day0_neural_results.jld2"),
     "results",
    (
        kernel = results_K,
        error = results_E,
        kernel_norm = results_Knorm,
        error_norm = results_Enorm,
        vexpl = results_vexpl

    )
)
