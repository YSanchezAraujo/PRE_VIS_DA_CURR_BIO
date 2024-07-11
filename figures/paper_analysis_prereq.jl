include("/Users/ysa/Documents/da_paper_code/fig/load_saved.jl");
cd("/Users/ysa/Documents/paper_figs")
using RobustModels
using AnovaBase;
using AnovaMixedModels;
using StatsBase;
using NaNMath;
using GLM;
for compartment in ["dms", "dls", "nacc", "errdms", "errdls", "errnacc"]
    for sub_comp in ["sr", "sl", "arc", "ari", "alc", "ali", "frc", "fri", "flc", "fli"]
        lrow = fill(NaN, 1, 50, 4)
        neu[43][compartment][sub_comp] = [neu[43][compartment][sub_comp]; lrow]
     end
end

function add_intercept(X)
    return [ones(size(X, 1), 1) X]
end

sder(X) = vec(nanstd(X, 1)) ./ sqrt(size(X, 1))

add_dim(x) = reshape(x, (size(x)..., 1))

drop_dim(a) = dropdims(a, dims = (findall(size(a) .== 1)...,))


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


mc = pyimport("matplotlib.colors")

mult_func(x) = x == 1 ? -1 : 1
l2dd(x) = mapslices(norm, x, dims=2) |> drop_dim
function nanl2norm(x)
    return sqrt(NaNMath.sum(Float64.(x).^2))
end

function beh_neu(neu, choice_weights, fips, reg, clab, event_map, final_day)
    neu_mat = reshape(
        hcat([l2dd(neu[fip][reg][event_map[reg_contra[reg][fip][clab]]][1:final_day, :, :]) for fip in fips]...),
        (final_day, 4, length(fips))
    )

    beh_mat = hcat(
        [choice_weights["mu"][fip][1:final_day, reg_contra[reg][fip][clab] + 1] *
        mult_func(reg_contra[reg][fip][clab]) for fip in fips]...
    )
    
    con_mod = neu_mat[:, 4, :] .- neu_mat[:, 1, :]
    
    return (
        cm=con_mod,
        all=neu_mat,
        beh=beh_mat
    )
end

function beh_neu_fb(neu, choice_weights, fips, reg, clab, event_map, final_day)
    fbfn(x) = mapslices(auc_trapz_pos_plus_neg, x; dims=2)
    #fbfn(x) = mapslices(norm, x; dims=2)
    neu_mat = reshape(
        hcat([fbfn(neu[fip][reg][event_map[reg_contra[reg][fip][clab]]][1:final_day, 5:15, :]) for fip in fips]...),
        (final_day, 4, length(fips))
    )

    beh_mat = hcat(
        [choice_weights["mu"][fip][1:final_day, reg_contra[reg][fip][clab] + 1] *
        mult_func(reg_contra[reg][fip][clab]) for fip in fips]...
    )
    
    con_mod = neu_mat[:, 1, :] .- neu_mat[:, 4, :]
    
    return (
        cm=con_mod,
        all=neu_mat,
        beh=beh_mat
    )
end



function neu_norms_day0(fip_res, fips, reg, clab)
    l2d(x) = mapslices(norm, x, dims=1) |> drop_dim
    neu_mat = hcat([l2d(fip_res[fip][reg][stim_map[reg_contra[reg][fip][clab]]]) for fip in fips]...)'
    con_mod = neu_mat[:, 4] .- neu_mat[:, 1]
    return (
        cm = con_mod, 
        all = neu_mat
    )
end

function bias_mult(bias)
    f = x -> x < 0 ? x * -1 : x
    pos_bias = f.(bias)
    rl = zeros(length(bias), 2)
    g = x -> x < 0 ? 1 : 2
    ind = g.(bias)
    rl[ind .== 1, 1] .= pos_bias[ind .== 1]
    rl[ind .== 2, 2] .= pos_bias[ind .== 2]
    return rl
end

nanmean(v, dim) = drop_dim(mapslices(NaNMath.mean, v; dims=dim))
nansem(v, dim) = drop_dim(mapslices(NaNMath.std, v; dims=dim)) ./ sqrt(size(v, dim))
nansum(v, dim) = drop_dim(mapslices(NaNMath.sum, v; dims=dim))

function zscore_transform(u)
    dt = fit(StatsBase.ZScoreTransform, u)
    return StatsBase.transform(dt, u)
end


function pval_perm(null_dist, ref)
    n = length(null_dist)

    if ref > 0
        p = sum(null_dist .> ref) / n
    end

    if ref < 0
        p = sum(null_dist .< ref) / n
    end

    return p
end

function binary_union_not_nan(x, y)
    x_nan_ind = findall(isnan.(x))
    y_nan_ind = findall(isnan.(y))
    nan_ind = union(x_nan_ind, y_nan_ind)
    return setdiff(1:length(x), nan_ind)
end

union_not_nan(l_vec) = setdiff(1:length(l_vec[1]), union([findall(isnan.(l)) for l in l_vec]...))

function non_nan_mice(x)
    any_nans = [any(isnan.(x[:, i])) for i in 1:size(x, 2)]
    return x[:, .!any_nans]
end

function nancor(x, y)
    use_ind = binary_union_not_nan(x, y)
    return cor(x[use_ind], y[use_ind])
end

function nancrosscor(x, y; ccrng=nothing)
    use_ind = binary_union_not_nan(x, y)
    if isnothing(ccrng)
        return crosscor(x[use_ind], y[use_ind])
    end
    return crosscor(x[use_ind], y[use_ind], ccrng)
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
        model = model,
        pred = predict(model)
    )
end

function binary_nan_robust_lm(x, y; ztransform=true)
    use_ind = binary_union_not_nan(x, y)

    if ztransform
        y_use = zscore_transform(y[use_ind])
        x_use = zscore_transform(x[use_ind])
    else
        y_use = y[use_ind]
        x_use = x[use_ind]
    end

    rlmdf = DataFrame(dep_var=y_use, indep_var=x_use);

"""
MEstimator{HuberLoss}(); 
method=:chol, 
initial_scale=10.0,
correct_leverage=true,
maxiter=10000,
"""
    
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
        p = pval,
        b0 = b0,
        beta = beta,
        r2 = r2_api,
        cor = cor_api,
        model = robust_model,
        pred = predict(robust_model)
        )
end


function robust_lm(x, y; ztransform=true)
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
    adjr2_api = StatsBase.adjr2(robust_model, :devianceratio)
    cor_api = NaN
    adjcor_api = NaN
    
    try
        cor_api = sign(beta) * sqrt(r2_api)
    catch 
        println("could not compute cor-coef")
        println(string("R2: ", round(r2_api; digits=3)))
    end

        
    return (
        p = pval,
        b0 = b0,
        beta = beta,
        pred = predict(robust_model),
        x = x_use,
        y = y_use,
        model = robust_model,
        cor_api = cor_api,
        cor_pear = nancor(y_use, x_use),
        adjcor_api = adjcor_api,
        r2_api = r2_api,
        adjr2_api = adjr2_api,
        )
end



function cor_beh_neu_pva_behday(beh_vals, neu_vals, day)
    xfc = beh_vals[day, :]
    cors = [nancor(xfc, neu_vals[d, :]) for d in 1:size(neu_vals, 1)]
    pvals = zeros(15)
    
    for d in 1:15
        model_d = robust_nanlm(neu_vals[d, :], xfc)
        pvals[d] = model_d.p
    end

    return (cor=cors, pval=pvals)
end

function cor_beh_neu_pval_behrngs(beh_vals, neu_vals, beh_rng)
    xfc = nanmean(beh_vals[beh_rng, :], 1)
    cors = [nancor(xfc, neu_vals[d, :]) for d in 1:size(neu_vals, 1)]    
    pvals = zeros(15)
    
    for d in 1:15
        model_d = robust_nanlm(neu_vals[d, :], xfc)
        pvals[d] = model_d.p
    end

    return (cor=cors, pval=pvals)
end

function pseudocor_beh_neu_pval_behrngs(beh_vals, neu_vals, beh_rng)
    xfc = nanmean(beh_vals[beh_rng, :], 1)
    cors = [robust_nanlm(neu_vals[d, :], xfc).cor_api for d in 1:size(neu_vals, 1)]    
    pvals = zeros(15)
    
    for d in 1:15
        model_d = robust_nanlm(neu_vals[d, :], xfc)
        pvals[d] = model_d.p
    end

    return (cor=cors, pval=pvals)
end


inds_all = [true for fip in fips]
use_inds = inds_all
final_day = 20
contra_dms = beh_neu(neu, choice_weights, fips[use_inds], "dms", "contra", stim_map, final_day);
ipsi_dms = beh_neu(neu, choice_weights, fips[use_inds], "dms", "ipsi", stim_map, final_day);
contra_nacc = beh_neu(neu, choice_weights, fips[use_inds], "nacc", "contra", stim_map, final_day);
ipsi_nacc = beh_neu(neu, choice_weights, fips[use_inds], "nacc", "ipsi", stim_map, final_day);
contra_dls = beh_neu(neu, choice_weights, fips[use_inds], "dls", "contra", stim_map, final_day);
ipsi_dls = beh_neu(neu, choice_weights, fips[use_inds], "dls", "ipsi", stim_map, final_day);

act_contra_dms = beh_neu(neu, choice_weights, fips[use_inds], "dms", "contra", actc_map, final_day);
act_ipsi_dms = beh_neu(neu, choice_weights, fips[use_inds], "dms", "ipsi", actc_map, final_day);
act_contra_nacc = beh_neu(neu, choice_weights, fips[use_inds], "nacc", "contra", actc_map, final_day);
act_ipsi_nacc = beh_neu(neu, choice_weights, fips[use_inds], "nacc", "ipsi", actc_map, final_day);
act_contra_dls = beh_neu(neu, choice_weights, fips[use_inds], "dls", "contra", actc_map, final_day);
act_ipsi_dls = beh_neu(neu, choice_weights, fips[use_inds], "dls", "ipsi", actc_map, final_day);

contra_dms_fb = beh_neu_fb(neu, choice_weights, fips[use_inds], "dms", "contra", fc_map, final_day);
ipsi_dms_fb = beh_neu_fb(neu, choice_weights, fips[use_inds], "dms", "ipsi", fc_map, final_day);
contra_nacc_fb = beh_neu_fb(neu, choice_weights, fips[use_inds], "nacc", "contra", fc_map, final_day);
ipsi_nacc_fb = beh_neu_fb(neu, choice_weights, fips[use_inds], "nacc", "ipsi", fc_map, final_day);
contra_dls_fb = beh_neu_fb(neu, choice_weights, fips[use_inds], "dls", "contra", fc_map, final_day);
ipsi_dls_fb = beh_neu_fb(neu, choice_weights, fips[use_inds], "dls", "ipsi", fc_map, final_day);

qc = load(joinpath("/Users/ysa/Documents/da_paper_code/data", "qcmaps.jld2"), "maps");

for (k, fip) in enumerate([collect(13:16); collect(26:43)])
    last_day = size(qc.qc[fip], 1) >= 20 ? 20 : size(qc.qc[fip], 1)
    
    ii_nacc = findall(.!qc.qc[fip][1:last_day, 1])
    ii_dms = findall(.!qc.qc[fip][1:last_day, 2])
    ii_dls = findall(.!qc.qc[fip][1:last_day, 3])

    contra_nacc.cm[ii_nacc, k] .= NaN
    ipsi_nacc.cm[ii_nacc, k] .= NaN

    contra_dms.cm[ii_dms, k] .= NaN
    ipsi_dms.cm[ii_dms, k] .= NaN

    contra_dls.cm[ii_dls, k] .= NaN
    ipsi_dls.cm[ii_dls, k] .= NaN

    contra_nacc.all[ii_nacc, :, k] .= NaN
    ipsi_nacc.all[ii_nacc, :, k] .= NaN

    contra_dms.all[ii_dms, :, k] .= NaN
    ipsi_dms.all[ii_dms, :, k] .= NaN

    contra_dls.all[ii_dls, :, k] .= NaN
    ipsi_dls.all[ii_dls, :, k] .= NaN
end


fip_res_d0 = load(joinpath("/Users/ysa/Documents/da_paper_code/data", "neu_day0.jld2"), "res");

day0_contra_dms = neu_norms_day0(fip_res_d0, 26:43, "dms", "contra");
day0_ipsi_dms = neu_norms_day0(fip_res_d0, 26:43, "dms", "ipsi");


day0_contra_dls = neu_norms_day0(fip_res_d0, 26:43, "dls", "contra");
day0_ipsi_dls = neu_norms_day0(fip_res_d0, 26:43, "dls", "ipsi");


day0_contra_nacc = neu_norms_day0(fip_res_d0, 26:43, "nacc", "contra");
day0_ipsi_nacc = neu_norms_day0(fip_res_d0, 26:43, "nacc", "ipsi");

qc_d0 = load(joinpath("/Users/ysa/Documents/da_paper_code/data", "qcd0.jld2"), "qc");


day0_contra_nacc.cm[.!qc_d0[:, 1]] .= NaN
day0_ipsi_nacc.cm[.!qc_d0[:, 1]] .= NaN
day0_contra_nacc.all[.!qc_d0[:, 1], :] .= NaN
day0_ipsi_nacc.all[.!qc_d0[:, 1], :] .= NaN

day0_contra_dms.cm[.!qc_d0[:, 2]] .= NaN
day0_ipsi_dms.cm[.!qc_d0[:, 2]] .= NaN
day0_contra_dms.all[.!qc_d0[:, 2], :] .= NaN
day0_ipsi_dms.all[.!qc_d0[:, 2], :] .= NaN

day0_contra_dls.cm[.!qc_d0[:, 3]] .= NaN
day0_ipsi_dls.cm[.!qc_d0[:, 3]] .= NaN
day0_contra_dls.all[.!qc_d0[:, 3], :] .= NaN
day0_ipsi_dls.all[.!qc_d0[:, 3], :] .= NaN

using NPZ

d0_trial_path = "/Users/ysa/Documents/da_paper_code/data"
contra_d0_dms = load(joinpath(d0_trial_path, "tcontra.jld2"), "trial")
ipsi_d0_dms = load(joinpath(d0_trial_path, "tipsi.jld2"), "trial");


contra_dms_fb = beh_neu_fb(neu, choice_weights, fips[use_inds], "dms", "contra", fc_map, final_day);
ipsi_dms_fb = beh_neu_fb(neu, choice_weights, fips[use_inds], "dms", "ipsi", fc_map, final_day);

contra_dls_fb = beh_neu_fb(neu, choice_weights, fips[use_inds], "dls", "contra", fc_map, final_day);
ipsi_dls_fb = beh_neu_fb(neu, choice_weights, fips[use_inds], "dls", "ipsi", fc_map, final_day);

contra_nacc_fb = beh_neu_fb(neu, choice_weights, fips[use_inds], "nacc", "contra", fc_map, final_day);
ipsi_nacc_fb = beh_neu_fb(neu, choice_weights, fips[use_inds], "nacc", "ipsi", fc_map, final_day);

nansem_1d(x) = std(x) / sqrt(length(x))
bias_beh = [choice_weights["mu"][fip][1:20, 1] for fip in fips]
bias_per_mouse = [bias_mult(bias_beh[j]) for j in 1:22];

contra_ind_fips = [reg_contra["dms"][f]["contra"] for f in fips]
contra_fx(x) = x == 1 ? -1 : 1
contra_mult_fips = contra_fx.(contra_ind_fips)
bias_contra_no_mult = hcat([choice_weights["mu"][f][1:20, 1] for f in fips]...)
bias_contra_mult = bias_contra_no_mult .* contra_mult_fips'

cifunc(u, ci) = percentile(u, [(100-ci)/2, 100-(100-ci)/2])

mult_func(x) = x == 1 ? -1 : 1


function bias_mult(bias)
    f = x -> x < 0 ? x * -1 : x
    pos_bias = f.(bias)
    rl = zeros(length(bias), 2)
    g = x -> x < 0 ? 1 : 2
    ind = g.(bias)
    rl[ind .== 1, 1] .= pos_bias[ind .== 1]
    rl[ind .== 2, 2] .= pos_bias[ind .== 2]
    return rl
end

bias_per_mouse = [bias_mult(bias_beh[j]) for j in 1:22];

function get_contra_ipsi_bias(bias_per_mouse, reg, fipgrp)
    c_idx = [reg_contra[reg][fip]["contra"] for fip in fipgrp]
    i_idx = [reg_contra[reg][fip]["ipsi"] for fip in fipgrp]

    contra_bias = [bias_per_mouse[i][:, c_i] for (i, c_i) in enumerate(c_idx)]
    ipsi_bias = [bias_per_mouse[i][:, i_i] for (i, i_i) in enumerate(i_idx)]

    return (
        contra = hcat(contra_bias...),
        ipsi = hcat(ipsi_bias...)
    )
end

function get_contra_ipsi_acc(cstat, fipgrp, reg)
    contra_side = [reg_contra[reg][fip]["contra"] for fip in fipgrp]
    ipsi_side = [reg_contra[reg][fip]["ipsi"] for fip in fipgrp]
    contra_acc = hcat([[cstat[fip].rightacc cstat[fip].leftacc][1:18, cside] for (cside, fip) in zip(contra_side, fipgrp)]...)
    ipsi_acc = hcat([[cstat[fip].rightacc cstat[fip].leftacc][1:18, iside] for (iside, fip) in zip(ipsi_side, fipgrp)]...)
    return (
        contra = contra_acc, 
        ipsi = ipsi_acc
    )
end

nonnan(x) = x[.!isnan.(x)]


function df_for_stats(model_weight, day0_neural, fip_grp, side_lab)
    mod_w = vec(model_weight)
    day = vec(hcat([collect(1:20) for f in fip_grp]...))
    fip_id = vec(hcat([ones(Int64, 20) * f for f in fip_grp]...))
    
    if typeof(day0_neural[1]) == String
        day0 = vec(hcat([repeat([d0v, 20]) for d0v in day0_neural]...))
    else
        day0 = vec(hcat([ones(20) * d0v for d0v in day0_neural]...))
    end    

    daysplits = split_day(day)

    # need to remove all nans and write function to do so
    use_inds = union_not_nan([mod_w, day, fip_id, day0])

    return DataFrame(
            contrast_weight = mod_w[use_inds],
            day = day[use_inds],
            mouse = fip_id[use_inds],
            neural_strength = day0[use_inds],
            day_split2 = daysplits.d2[use_inds],
            day_split3 = daysplits.d3[use_inds],
            day_split4 = daysplits.d4[use_inds],
        )
end

function df_for_stats_bias_chist(model_weight, day0_vals, fip_grp, side_lab)
    mod_w = vec(model_weight)
    day = vec(hcat([collect(1:20) for f in fip_grp]...))
    fip_id = vec(hcat([ones(Int64, 20) * f for f in fip_grp]...))
    
    if typeof(day0_vals[1]) == String
        day0 = vec(hcat([repeat([d0v, 20]) for d0v in day0_vals]...))
    else
        day0 = vec(hcat([ones(20) * d0v for d0v in day0_vals]...))
    end

    daysplits = split_day(day)
    
    # need to remove all nans and write function to do so
    use_inds = union_not_nan([mod_w, day, fip_id, day0])

    return DataFrame(
            dep_weight = mod_w[use_inds],
            day = day[use_inds],
            mouse = fip_id[use_inds],
            d0_strength = day0[use_inds],
            day_split2 = daysplits.d2[use_inds],
            day_split3 = daysplits.d3[use_inds],
            day_split4 = daysplits.d4[use_inds],
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

dms_w_cor_c = [nancor(contra_dms.cm[:, j], contra_dms.beh[:, j]) for j in 1:22]
dls_w_cor_c = [nancor(contra_dls.cm[:, j], contra_dls.beh[:, j]) for j in 1:22]
nacc_w_cor_c = [nancor(contra_nacc.cm[:, j], contra_nacc.beh[:, j]) for j in 1:22];

dms_w_cor_i = [nancor(ipsi_dms.cm[:, j], ipsi_dms.beh[:, j]) for j in 1:22]
dls_w_cor_i = [nancor(ipsi_dls.cm[:, j], ipsi_dls.beh[:, j]) for j in 1:22]
nacc_w_cor_i = [nancor(ipsi_nacc.cm[:, j], ipsi_nacc.beh[:, j]) for j in 1:22];

chist = hcat([choice_weights["mu"][f][1:20, 4] for f in fips]...)
