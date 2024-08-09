import NaNMath;
using LinearAlgebra;
using PyCall;
using SparseArrays;

mpl = pyimport("matplotlib.lines");

function add_intercept(X)
    return [ones(size(X, 1), 1) X]
end

function sample_rate(x::Vector)
    (x[end] - x[1]) / length(x)
end

non_zero(x) = x[x .!= 0]

non_val(x, v) = x[x .!= v]

add_dim(x) = reshape(x, (size(x)..., 1))

_drop_dim(a) = dropdims(a, dims = (findall(size(a) .== 1)...,))

function drop_dim(a)
    nm = size(a)

    if length(nm) > 1
        if (nm[1] == 1) && (nm[2] == 1)
            return a[1, 1]
        else
            _drop_dim(a)
        end
    else
        return a
    end
end

nanmean(x) = NaNMath.mean(x)

function nanmean(X, dim)
    return drop_dim(mapslices(NaNMath.mean, X, dims=dim))
end

nanstd(x) = NaNMath.std(x)

function nanstd(X, dim)
    return drop_dim(mapslices(NaNMath.std, X, dims=dim))
end

function split_inds(arr, n::Int64)
    return Vector.(collect(Iterators.partition(arr, n)))
end

function weights_by_event(W, n_stim,  event_names)
    weights = Dict()
    event_inds = split_inds(1:size(W, 2), n_stim)
    for (i, event_name) in enumerate(event_names)
        weights[event_name] = W[:, event_inds[i]]
    end
    return weights
end


st = pyimport("scipy.interpolate")
function resample_data(times, values; rate=0.02, return_time=false, kind="linear")
    interp_fn_data = st.interp1d(times, values, kind=kind)
    regular_grid = minimum(times):rate:(maximum(times))
    if return_time
        return [regular_grid interp_fn_data(regular_grid)]
    end
    return interp_fn_data(regular_grid)
end


function exp_time_filt(y, alpha)
    n = length(y)
    filt = zeros(n)
    x = zeros(Float64, n)

    for j in 1:n
        if y[j] == 1
            x[j] = 1
        elseif y[j] == 0
            x[j] = -1
        end
    end

    filt[1] = x[1]

    for j in 2:n
        filt[j] = alpha * x[j] + (1 - alpha) * filt[j - 1]
    end


    return [0.; filt[1:n-1]]
end

_safe_index(x, i) = length(x[i]) == 0 ? 0 : x[i]

np = pyimport("numpy");

_fake_bools(a) = a > 0 ? true : false

function find_roots(x, y)
    s = _fake_bools.(abs.(diff(sign.(y))))
    z = x[1:end-1][s] .+ diff(x)[s] ./ (abs.(y[2:end][s] ./ y[1:end-1][s]) .+ 1)
    return z
end

function auc_trapz_pos(y; x=nothing, dx=1/10)
    x = isnothing(x) ? collect(1:length(y)) .* dx : dx
    xz = find_roots(x, y)

    x2 = [x; xz]
    y2 = [y; zeros(length(xz))]

    k = sortperm(x2)

    x2 = x2[k]
    y2 = y2[k]

    pos_y2 = max.(y2, 0.0)

    return np.trapz(pos_y2, x2)
end

function auc_trapz_neg(y; x=nothing, dx=1/10)
    y_flipped = y .* -1
    return -auc_trapz_pos(y_flipped; x=x, dx=dx)
end

function auc_trapz_pos_plus_neg(y; x=nothing, dx=1/10)
    auc_pos = auc_trapz_pos(y; x=x, dx=dx)
    auc_neg = auc_trapz_neg(y; x=x, dx=dx)
    return auc_pos + auc_neg
end



"""
these functions are used to load the data from STAN files, 
might be deprecated
""";
pd = pyimport("pandas")

function collect_estimates_matrix(base_path, file_name, param_name, iter_j, iter_k, idx_col)
    load_file = joinpath(base_path, file_name)
    data = pd.read_csv(load_file, comment="#", index_col=idx_col)

    mean_est = zeros((iter_j, iter_k))
    five_p = similar(mean_est)
    fifty_p = similar(mean_est)
    ninefive_p = similar(mean_est)
    mcse = similar(mean_est)

    for j in 1:iter_j
        for k in 1:iter_k
            name = string(param_name, "[", j, ",", k, "]")
            row_data = get(data.loc, (name))

            mean_est[j, k] = row_data.Mean
            five_p[j, k] = row_data[4]
            fifty_p[j, k] = row_data[5]
            ninefive_p[j, k] = row_data[6]
            mcse[j, k] = row_data.MCSE
 
        end
    end

    return mean_est, five_p, fifty_p, ninefive_p, mcse
end

function collect_estimates_vector(base_path, file_name, param_name, iter_j, idx_col)
    load_file = joinpath(base_path, file_name)
    data = pd.read_csv(load_file, comment="#", index_col=idx_col)

    mean_est = zeros(iter_j)
    five_p = similar(mean_est)
    fifty_p = similar(mean_est)
    ninefive_p = similar(mean_est)
    mcse = similar(mean_est)


    for j in 1:iter_j
        name = string(param_name, "[", j, "]")
        row_data = get(data.loc, (name))

        mean_est[j] = row_data.Mean
        five_p[j] = row_data[4]
        fifty_p[j] = row_data[5]
        ninefive_p[j] = row_data[6]
        mcse[j] = row_data.MCSE

    end

    return mean_est, five_p, fifty_p, ninefive_p, mcse
end

function collect_estimates_real(base_path, file_name, param_name, idx_col)
    load_file = joinpath(base_path, file_name)
    data = pd.read_csv(load_file, comment="#", index_col=idx_col)

    mean_est = 0.0
    five_p = 0.0
    fifty_p = 0.0
    ninefive_p = 0.0
    mcse = 0.0
    
    row_data = get(data.loc, param_name)

    mean_est = row_data.Mean
    five_p = row_data[4]
    fifty_p = row_data[5]
    ninefive_p = row_data[6]
    mcse = row_data.MCSE

    return mean_est, five_p, fifty_p, ninefive_p, mcse
end


dms_contra_map = Dict(
    13 => "stim_left",
    14 => "stim_left",
    15 => "stim_left",
    16 => "stim_left",
    26 => "stim_right",
    27 => "stim_right",
    28 => "stim_right",
    29 => "stim_right",
    30 => "stim_left",
    31 => "stim_left",
    32 => "stim_left",
    33 => "stim_left",
    34 => "stim_left",
    35 => "stim_left",
    36 => "stim_left",
    37 => "stim_right",
    38 => "stim_right",
    39 => "stim_left",
    40 => "stim_right",
    41 => "stim_right",
    42 => "stim_left",
    43 => "stim_left", 
)

dms_ipsi_map = Dict(
    13 => "stim_right",
    14 => "stim_right",
    15 => "stim_right",
    16 => "stim_right",
    26 => "stim_left",
    27 => "stim_left",
    28 => "stim_left",
    29 => "stim_left",
    30 => "stim_right",
    31 => "stim_right",
    32 => "stim_right",
    33 => "stim_right",
    34 => "stim_right",
    35 => "stim_right",
    36 => "stim_right",
    37 => "stim_left",
    38 => "stim_left",
    39 => "stim_right",
    40 => "stim_left",
    41 => "stim_left",
    42 => "stim_right",
    43 => "stim_right", 
)


function true_psychometric(choice, contrast, coh)
    coh_idx = contrast .== coh

    coh_N = sum(coh_idx)

    if coh_N == 0
        return (
            p = NaN, 
            err = NaN
        )
    end

    p = sum(choice[coh_idx] .== 0) / coh_N

    p_err = sqrt(p * (1 - p) / coh_N)

    return (
        p = p, 
        err = p_err
    )
end

function pred_psychometric(probs, signcon, coh)
    coh_idx = signcon .== coh

    coh_N = sum(coh_idx)

    if coh_N == 0
        return (
            p = NaN, 
            err = NaN
        )
    end

    p = mean(probs[coh_idx])

    p_err = sqrt(p * (1 - p) / coh_N)

    return (
        p = p, 
        err = p_err
    )
end

function transform_X(x_df, day)

    choice_map = Dict(1.0 => 1, -1.0 => 0, 0.0 => NaN)

    n = size(x_df, 1)

    _X = [x_df.cright x_df.cleft]

    ses = ones(Int64, size(_X, 1)) .* day

    return (x = _X, 
            y = [choice_map[xi] for xi in x_df.choice], 
            sc = x_df.signed_contrast, 
            ses=ses)
end

function choice_kern(choice, alpha)
    choice_use = choice[choice .!= 0]

    N = length(choice_use)

    choice_kernel = zeros(N)

    for t in 1:N-1
        choice_kernel[t+1] = choice_kernel[t] + alpha * (choice_use[t]- choice_kernel[t])
    end

    return choice_kernel

end

function average_signal(signal, inds, window)

    n_trials = size(inds, 1)

    out = fill(NaN, n_trials, window)

    max_ind = length(signal)

    for (t, (start, stop)) in enumerate(inds)
        if stop > max_ind
            out[t, 1:length(start:max_ind)] = signal[start:max_ind]
        else
            out[t, :] = signal[start:stop]
        end
    end

    avg = vec(mean(out; dims=1))

    return (
        mean = avg,
        trial = out
    )
end

function rsquared(y, yhat)
    y_bar = mean(y)
    quo = sum((y .- yhat).^2) / sum((y .- y_bar).^2)
    return 1 - quo
end