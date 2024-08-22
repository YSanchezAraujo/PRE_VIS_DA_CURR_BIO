include("utility.jl");

function make_cosine_basis_linear(nB, peak_range, dt, time_range)
    raisedCosfn(x, c, dc) = (cos(max(-pi,min(pi,(x-c)*pi/dc/2)))+1)/2
    dCtr = (diff(peak_range) / (nB - 1))[1]
    Bctrs = peak_range[1]:dCtr:peak_range[2]
    basisPkeas = Bctrs
    tgrid = time_range[1]:dt:time_range[end]
    nT = length(tgrid)
    cosbasis = raisedCosfn.(repeat(tgrid, 1, nB), repeat(Bctrs', nT, 1), dCtr)
    return cosbasis
end


GEN_BAS_N = 16
function gen_basis()
    n_cos_fn = GEN_BAS_N

    basis_params = (
        n_vectors = n_cos_fn,
        peak_range = [1, 50],
        dt = 1,
        time_range = 1:1:50
    )

    return make_cosine_basis_linear(basis_params.n_vectors, 
                        basis_params.peak_range, 
                        basis_params.dt, 
                        basis_params.time_range)
end

function gen_basis(params)
    n_cos_fn = Int64(params[1])

    basis_params = (
        n_vectors = n_cos_fn,
        peak_range = params[2:3],
        dt = params[4],
        time_range = 1:1:params[3]
    )

    return make_cosine_basis_linear(basis_params.n_vectors, 
                        basis_params.peak_range, 
                        basis_params.dt, 
                        basis_params.time_range)
end

function make_features(n_samples, start_indices, window)
    f0 = zeros(n_samples)
    f0[start_indices] .= 1.0
    F = zeros(n_samples, window)
    F[:, 1] = f0
    
    for f_i in 2:window
        F[:, f_i] = circshift(f0, f_i-1)
    end
    
    return F
end

"""
data is the named tuple from the mouse_session_data function defined in the preprocess file
"""
function make_design_matrix_items(data, window::Int64, nfunc::Int64; acausal_action_lag=0)
    n_samples = size(data.neural, 1)
    convals = [0.0625, 0.125, 0.25, 1.]
    start_peaks, end_peaks, dt = 1, window, 1

    basis = gen_basis([nfunc, start_peaks, end_peaks, dt])

    stim_right = sparse.([
        make_features(n_samples, 
                      data.stim_idx[data.behavior.contrast_right .== v], 
                      window) * basis for v in convals
    ])

    stim_left = sparse.([
        make_features(n_samples, 
                      data.stim_idx[data.behavior.contrast_left .== v], 
                      window) * basis for v in convals
    ])

    act_right_correct = sparse.([
        make_features(n_samples, 
                      data.act_idx[(data.behavior.contrast_right .== v) .& (data.behavior.reward_type .== 1)] .- acausal_action_lag, 
                      window) * basis for v in convals
    ])

    act_right_incorrect = sparse.([
        make_features(n_samples, 
                      data.act_idx[(data.behavior.contrast_right .== v) .& (data.behavior.reward_type .== -1)] .- acausal_action_lag, 
                      window) * basis for v in convals
    ])

    act_left_correct = sparse.([
        make_features(n_samples, 
                      data.act_idx[(data.behavior.contrast_left .== v) .& (data.behavior.reward_type .== 1)] .- acausal_action_lag, 
                      window) * basis for v in convals
    ])

    act_left_incorrect = sparse.([
        make_features(n_samples, 
                      data.act_idx[(data.behavior.contrast_left .== v) .& (data.behavior.reward_type .== -1)] .- acausal_action_lag, 
                      window) * basis for v in convals
    ])

    reward_right_correct = sparse.([
        make_features(n_samples, 
                      data.reward_idx[(data.behavior.contrast_right .== v) .& (data.behavior.reward_type .== 1)], 
                      window) * basis for v in convals
    ])

    reward_right_incorrect = sparse.([
        make_features(n_samples, 
                      data.reward_idx[(data.behavior.contrast_right .== v) .& (data.behavior.reward_type .== -1)], 
                      window) * basis for v in convals
    ])

    reward_left_correct = sparse.([
        make_features(n_samples, 
                      data.reward_idx[(data.behavior.contrast_left .== v) .& (data.behavior.reward_type .== 1)], 
                      window) * basis for v in convals
    ])

    reward_left_incorrect = sparse.([
        make_features(n_samples, 
                      data.reward_idx[(data.behavior.contrast_left .== v) .& (data.behavior.reward_type .== -1)], 
                      window) * basis for v in convals
    ])


    return (stim_right = stim_right, 
            stim_left = stim_left, 
            act_right_correct = act_right_correct,
            act_right_incorrect = act_right_incorrect,
            act_left_correct = act_left_correct, 
            act_left_incorrect = act_left_incorrect,
            reward_right_correct = reward_right_correct, 
            reward_right_incorrect = reward_right_incorrect, 
            reward_left_correct = reward_left_correct, 
            reward_left_incorrect = reward_left_incorrect,
            basis = basis)
end

function design_matrix(design_matrix_items)
    SR = hcat(design_matrix_items.stim_right...)
    SL = hcat(design_matrix_items.stim_left...)
    ARC = hcat(design_matrix_items.act_right_correct...)
    ARI = hcat(design_matrix_items.act_right_incorrect...)
    ALC = hcat(design_matrix_items.act_left_correct...)
    ALI = hcat(design_matrix_items.act_left_incorrect...)
    RRC = hcat(design_matrix_items.reward_right_correct...)
    RRI = hcat(design_matrix_items.reward_right_incorrect...)
    RLC = hcat(design_matrix_items.reward_left_correct...)
    RLI = hcat(design_matrix_items.reward_left_incorrect...)
    return add_intercept([SR SL ARC ARI ALC ALI RRC RRI RLC RLI])
end

function truncate_design_matrix(desmat, data, window)
    z = collect(zip(data.stim_idx[1:end-1], data.reward_idx[1:end-1] .+ (window - 1)))
    z_final_stim = data.stim_idx[end]
    z_final_reward = data.reward_idx[end] .+ (window - 1)
    z_final_reward = z_final_reward > size(data.neural, 1) ? size(data.neural, 1) : z_final_reward

    push!(z, (z_final_stim, z_final_reward))

    truncated_desmat = []
    truncated_Y = []

    # Calculate new indices for the events in the truncated vectors
    new_stim_indices = []
    new_act_indices = []
    new_reward_indices = []

    cumulative_length = 0

    for i in 1:length(z)
        start_idx, end_idx = z[i]

        push!(truncated_desmat, desmat[start_idx:end_idx, :])
        push!(truncated_Y, data.neural[start_idx:end_idx, :])

        length_segment = end_idx - start_idx + 1

        # calculate new indices
        new_stim_idx = data.stim_idx[(data.stim_idx .>= start_idx) .& (data.stim_idx .<= end_idx)] .- start_idx .+ 1 .+ cumulative_length
        new_reward_idx = data.reward_idx[(data.reward_idx .>= start_idx) .& (data.reward_idx .<= end_idx)] .- start_idx .+ 1 .+ cumulative_length
        new_act_idx = data.act_idx[(data.act_idx .>= start_idx) .& (data.act_idx .<= end_idx)] .- start_idx .+ 1 .+ cumulative_length

        append!(new_stim_indices, new_stim_idx)
        append!(new_act_indices, new_act_idx)
        append!(new_reward_indices, new_reward_idx)

        cumulative_length += length_segment
    end

    return (
        X = truncated_desmat, 
        Y = truncated_Y, 
        stim_idx = new_stim_indices,
        act_idx = new_act_indices,
        reward_idx = new_reward_indices
        )
end

"""
from this point on the code is for day 0 (pre-exposure) result
""";
function make_design_matrix_items_day0(data, window::Int64, nfunc::Int64; acausal_action_lag=0)
    n_samples = size(data.neural, 1)
    convals = [0.0625, 0.125, 0.25, 1.]
    start_peaks, end_peaks, dt = 1, window, 1

    basis = gen_basis([nfunc, start_peaks, end_peaks, dt])

    stim_right = sparse.([
        make_features(n_samples, 
                      data.stim_idx[data.behavior.contrast_right .== v], 
                      window) * basis for v in convals
    ])

    stim_left = sparse.([
        make_features(n_samples, 
                      data.stim_idx[data.behavior.contrast_left .== v], 
                      window) * basis for v in convals
    ])

    return (stim_right = stim_right, 
            stim_left = stim_left,
            basis = basis)
end

function design_matrix_day0(design_matrix_items)
    SR = hcat(design_matrix_items.stim_right...)
    SL = hcat(design_matrix_items.stim_left...)
    return add_intercept([SR SL])
end


function truncate_design_matrix_day0(desmat, data, window)
    z = collect(zip(data.stim_idx[1:end-1], data.stim_idx[1:end-1] .+ (window - 1)))
    z_final_stim = data.stim_idx[end]
    z_final_stim_end = data.stim_idx[end] .+ (window - 1)
    z_final_stim_end = z_final_stim_end > size(data.neural, 1) ? size(data.neural, 1) : z_final_stim_end

    push!(z, (z_final_stim, z_final_stim_end))

    truncated_desmat = []
    truncated_Y = []

    # Calculate new indices for the events in the truncated vectors
    new_stim_indices = []
    cumulative_length = 0

    for i in 1:length(z)
        start_idx, end_idx = z[i]

        push!(truncated_desmat, desmat[start_idx:end_idx, :])
        push!(truncated_Y, data.neural[start_idx:end_idx, :])

        length_segment = end_idx - start_idx + 1

        # calculate new indices
        new_stim_idx = data.stim_idx[(data.stim_idx .>= start_idx) .& (data.stim_idx .<= end_idx)] .- start_idx .+ 1 .+ cumulative_length
        
        append!(new_stim_indices, new_stim_idx)
        
        cumulative_length += length_segment
    end

    return (
        X = truncated_desmat, 
        Y = truncated_Y, 
        stim_idx = new_stim_indices,
        )
end
