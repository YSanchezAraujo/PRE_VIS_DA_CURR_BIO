import NaNMath;
using LinearAlgebra;
using PyCall;
using SparseArrays;
using NPZ;
using JLD2;
using StatsBase;
using DataFrames;
using RobustModels;
using GLM;

mutable struct MouseBehavior
    avg::Matrix
    ci275::Matrix
    ci975::Matrix
end

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


nansem(X, dim) = drop_dim(nanstd(X, dim)) ./ sqrt(size(X, dim))

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

function binary_union_not_nan(x, y)
    x_nan_ind = findall(isnan.(x))
    y_nan_ind = findall(isnan.(y))
    nan_ind = union(x_nan_ind, y_nan_ind)
    return setdiff(1:length(x), nan_ind)
end

union_not_nan(l_vec) = setdiff(1:length(l_vec[1]), union([findall(isnan.(l)) for l in l_vec]...))

function nancor(x, y)
    use_ind = binary_union_not_nan(x, y)
    return cor(x[use_ind], y[use_ind])
end

function zscore_transform(u)
    dt = fit(StatsBase.ZScoreTransform, u)
    return StatsBase.transform(dt, u)
end

function binary_nan_lm(x, y; ztransform=true)
    use_ind = binary_union_not_nan(x, y)

    if ztransform
        y_use = zscore_transform(y[use_ind])
        x_use = zscore_transform(x[use_ind])
    else
        y_use = y[use_ind]
        x_use = x[use_ind]
    end

    model = lm(Float64.(add_intercept(x_use)), Float64.(y_use))
    pval = coeftable(model).cols[4][2]
    b0, beta = coef(model)

    r2_val = StatsBase.r2(model, :devianceratio)
    cor_val = NaN

    try
        cor_val = sign(beta) * sqrt(r2_val)
    catch
        nothing
    end

    return (
        p = pval,
        b0 = b0,
        beta = beta,
        r2 = r2_val,
        cor = cor_val,
    )
end


function robust_nanlm(x, y; ztransform=true)
    use_ind = binary_union_not_nan(x, y)

    if ztransform
        y_use = zscore_transform(y[use_ind])
        x_use = zscore_transform(x[use_ind])
    else
        y_use = y[use_ind]
        x_use = x[use_ind]
    end

    rlmdf = DataFrame(dep_var=y_use, indep_var=x_use);

    robust_model = RobustModels.rlm(@formula(dep_var ~ 1 + indep_var), rlmdf,
        MEstimator{HuberLoss}(); 
        method=:chol, 
        initial_scale=10.0,
        correct_leverage=true,
        maxiter=10000,
    )

    pval = coeftable(robust_model).cols[4][2]
    b0, beta = coef(robust_model)

    r2_api = StatsBase.r2(robust_model, :devianceratio)
    cor_api = NaN
    
    try
        cor_api = sign(beta) * sqrt(r2_api)
    catch 
        nothing
    end

    return (
        b0 = b0,
        beta = beta,
        cor = cor_api,
        r2 = r2_api,
        pval = pval
    )
end

function get_qc(base_path, mouseid)
    session_paths = get_session_paths(base_path, mouseid)
    fx(x) = x == 0 ? false : true
    non_pretrain_session = fx.(session_paths.session)
    pretrain_session = .!non_pretrain_session
    
    return (
        train = session_paths[non_pretrain_session, [:session, :QC_NAcc, :QC_DMS, :QC_DLS]],
        pretrain = session_paths[pretrain_session, [:session, :QC_NAcc, :QC_DMS, :QC_DLS]]
        )
end


function split_day(days)
    f2(x) = (x > 0) && (x < 11) ? 1 : 2

    f3(x) = ((x > 0) && (x < 8)) ? 1 : ( ((x > 7) && (x < 15)) ? 2 : ( (x > 14) ? 3 : nothing  ) )

    f4(x) = (
        ((x > 0) && (x < 6)) ? 1 : ( 
            ((x > 5) && (x < 11)) ? 2 : ( ((x > 10) && ( x < 16)) ? 3 : ( (x > 15) ? 4 : nothing )  ) 
        )
    )

    return (
        d2 = f2.(days),
        d3 = f3.(days),
        d4 = f4.(days)
        )
end

function df_for_stats(beh_weights, neural_weights, mouse_ids; max_days=20)
    beh_vec = vec(beh_weights)
    day0_neu_vec = vec(hcat([ones(max_days) * d0v for d0v in neural_weights]...))
    day_vec = vec(hcat([1:max_days for f in mouse_ids]...))
    daysplits = split_day(day_vec)
    mouse_id_vec = vec(hcat([ones(Int64, max_days) * f for f in mouse_ids]...))
    use_inds = union_not_nan([beh_vec, day0_neu_vec, mouse_id_vec, day_vec])

    return DataFrame(
        contrast_weight = beh_vec[use_inds],
        day = day_vec[use_inds],
        mouse = mouse_id_vec[use_inds],
        neural_strength = day0_neu_vec[use_inds],
        day_split2 = daysplits.d2[use_inds],
        day_split3 = daysplits.d3[use_inds],
        day_split4 = daysplits.d4[use_inds],
    )
end