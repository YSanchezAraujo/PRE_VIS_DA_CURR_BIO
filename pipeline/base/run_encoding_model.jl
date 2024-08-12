include(joinpath(@__DIR__, "pipeline/base/encoding_model.jl"))
include(joinpath(@__DIR__, "pipeline/base/preprocess.jl"))
include(joinpath(@__DIR__, "pipeline/base/design_matrix.jl"))


function scalar_summary_stats(weights, event_names, func)
    summary = Dict()

    for ev in event_names
        setindex!(summary, mapslices(func, weights[ev]; dims=1), ev)
    end

    return summary
end

function fit_mouse_all_days(mouse_id, params, event_names, summary_func; max_sessions=20)
    nfunc, window, n_stim, n_sets, base_path = params.nfunc, params.window, params.n_stim, params.n_sets, params.base_path
    session_paths = get_session_paths(base_path, mouse_id)
    num_sessions = maximum(session_paths.session)
    fx(x) = x == 0 ? false : true
    non_pretrain_session = fx.(session_paths.session)
    session_has_neural = .!fx.(session_paths.neural_missing)
    regular_fit_sessions = session_paths.session[non_pretrain_session .& session_has_neural]

    kernel_estimates = Dict(
        "NAcc" => Dict(ev => fill(NaN, max_sessions, window, n_stim) for ev in event_names), 
        "DMS" => Dict(ev => fill(NaN, max_sessions, window, n_stim) for ev in event_names),
        "DLS" => Dict(ev => fill(NaN, max_sessions, window, n_stim) for ev in event_names)
    );

    error_estimates = Dict(
        "NAcc" => Dict(ev => fill(NaN, max_sessions, window, n_stim) for ev in event_names), 
        "DMS" => Dict(ev => fill(NaN, max_sessions, window, n_stim) for ev in event_names),
        "DLS" => Dict(ev => fill(NaN, max_sessions, window, n_stim) for ev in event_names)
    );

    kernel_estimates_summaries = Dict(
        "NAcc" => Dict(ev => fill(NaN, max_sessions, n_stim) for ev in event_names), 
        "DMS" => Dict(ev => fill(NaN, max_sessions, n_stim) for ev in event_names),
        "DLS" => Dict(ev => fill(NaN, max_sessions, n_stim) for ev in event_names)
    );

    error_estimates_summaries = Dict(
        "NAcc" => Dict(ev => fill(NaN, max_sessions, n_stim) for ev in event_names), 
        "DMS" => Dict(ev => fill(NaN, max_sessions,  n_stim) for ev in event_names),
        "DLS" => Dict(ev => fill(NaN, max_sessions,  n_stim) for ev in event_names)
    );

    var_expl = Dict(
        "NAcc" => fill(NaN, max_sessions), 
        "DMS" => fill(NaN, max_sessions),
        "DLS" => fill(NaN, max_sessions)
    )

    reg_labels = ["NAcc", "DMS", "DLS"]

    for day in regular_fit_sessions
        # load data and create design matrices
        data = mouse_session_data(base_path, mouse_id, day)
        desmat_items = make_design_matrix_items(data, window, nfunc)
        full_desmat = design_matrix(desmat_items)

        # this design matrix is truncated to only behavioral events 
        trunc_desmat_items = truncate_design_matrix(full_desmat, data, window);
        X = vcat(trunc_desmat_items.X...)
        Y = vcat(trunc_desmat_items.Y...)

        for (reg, rlabel) in enumerate(reg_labels) # 3 regions, NAcc, DMS, DLS in that order
            model_fit = bayes_ridge(X, zscore(Y[:, reg]))

            # extract the weights in the standard basis
            W = desmat_items.basis * reshape(model_fit.w[2:end], (nfunc, n_stim * n_sets)) # first index is the intercept
            S = desmat_items.basis * reshape(sqrt.(diag(model_fit.covar)[2:end]), (nfunc, n_stim * n_sets))

            # put the estimated model estimates in a less error prone data structure
            kernels = weights_by_event(W, n_stim, event_names)
            errors = weights_by_event(S, n_stim, event_names)

            #compute summary statistics of the kernels
            kernel_norms = scalar_summary_stats(kernels, event_names, summary_func)
            error_norms = scalar_summary_stats(errors, event_names, summary_func)

            # compute rsquared
            var_expl[rlabel][day] = rsquared(zscore(Y[:, reg]), X * model_fit.w)

            # store results
            for ev in event_names
                kernel_estimates[rlabel][ev][day, :, :] = kernels[ev]
                error_estimates[rlabel][ev][day, :, :] = errors[ev]

                kernel_estimates_summaries[rlabel][ev][day, :] = kernel_norms[ev]
                error_estimates_summaries[rlabel][ev][day, :] = error_norms[ev]
            end

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
    n_sets = 10, # this corresponds to the number of events in the design matrix, see event names below
    base_path = "/Users/ysa/Desktop/Subjects" # whereever you have downloaded the data
);

event_names = [
    "stim_right", "stim_left", 
    "act_right_correct", "act_right_incorrect",
    "act_left_correct", "act_left_incorrect", 
    "reward_right_correct", "reward_right_incorrect", 
    "reward_left_correct", "reward_left_incorrect"
];

mouse_ids = [collect(13:16); collect(26:43)];
results_K = Dict();
results_E = Dict();
results_Knorm = Dict();
results_Enorm = Dict();
results_vexpl = Dict();

for mouse in mouse_ids
    K, E, Knorm, Enorm, vexpl = fit_mouse_all_days(mouse, param_set, event_names, norm);
    setindex!(results_K, K, mouse)
    setindex!(results_E, E, mouse)
    setindex!(results_Knorm, Knorm, mouse)
    setindex!(results_Enorm, Enorm, mouse)
    setindex!(results_vexpl, vexpl, mouse)
end

save_path = "/Users/ysa/Desktop/pipeline/saved_results"

# save the results
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
