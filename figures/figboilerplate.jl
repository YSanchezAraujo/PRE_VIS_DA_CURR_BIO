using StatsBase
using Distributions
using LinearAlgebra
using PyCall
using SparseArrays
using JLD2
using PyPlot
using JSON
using Clustering
using DataFrames
PyDict(matplotlib["rcParams"])["font.size"] = 22
PyDict(matplotlib["rcParams"])["pdf.fonttype"] = Int64(42)
PyDict(matplotlib["rcParams"])["axes.spines.right"] = false
PyDict(matplotlib["rcParams"])["axes.spines.top"] = false
PyDict(matplotlib["rcParams"])["axes.linewidth"] = 2
PyDict(matplotlib["rcParams"])["figure.figsize"] = (5, 4)
import LogExpFunctions: logistic


np = pyimport("numpy")
sns = pyimport("seaborn")
pd = pyimport("pandas")

allfips = [collect(13:16); collect(26:43); collect(901:908); collect(910:914)]

function load_fip_beh(fip)
    beh_path = "/Users/ysa/Documents/da_paper_code/data/npyfiles/"
    y_str = string("fip_", fip, "_choice_y_orig_rtcheck_pc.npy")
    X_str = string("fip_", fip, "_choice_X_orig_rtcheck_pc.npy")
    r_str = string("fip_", fip, "_choice_rewards_rtcheck_pc.npy")
    acc_str = string("fip_", fip, "_choice_accuracy_rtcheck_pc.npy")
    sesmap_str = string("fip_", fip, "_choice_sesmap_orig_rtcheck_pc.npy")

    sesmap = npzread(joinpath(beh_path, sesmap_str))
    bools = [sesmap .== j for j in 1:maximum(sesmap)]
    bigx = npzread(joinpath(beh_path, X_str))
    bigy = npzread(joinpath(beh_path, y_str))
    bigr = npzread(joinpath(beh_path, r_str))
    acc = npzread(joinpath(beh_path, acc_str))

    x_days = [bigx[b, :] for b in bools]
    y_days = [bigy[b] for b in bools]
    r_days = [bigr[b] for b in bools]

    return (
        x = x_days,
        y = y_days,
        acc = acc,
        reward = r_days,
        sesmap = sesmap,
        ses_ind = bools
    )
end

function make_design_mat(X, y, alpha, cohalpha)
    _x = add_intercept(tanh.(cohalpha * X[:, [1, 2]]) ./ tanh(cohalpha))
    _x_ck = choice_kernel(alpha, y)
    return [_x _x_ck]
end

function x_standard()
    return [0 1; 0 0.25; 0 0.125; 0 0.0625; 0.0625 0; 0.125 0; 0.25 0; 1 0]
end


function probs(X, choice_weights, fip, day)
    w  = choice_weights["mu"][fip][day, :]
    return logistic.(X*w)
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

function true_psychometric(choice, contrast, coh)
    coh_idx = contrast .== coh

    coh_N = sum(coh_idx)

    if coh_N == 0
        return (
            p = NaN, 
            err = NaN
        )
    end

    p = sum(choice[coh_idx]) / coh_N

    p_err = sqrt(p * (1 - p) / coh_N)

    return (
        p = p, 
        err = p_err
    )
end


function load_stan_res(path, fip)
    base_path = joinpath(path, "fip$fip")
    files = readdir(base_path)
    match_str = "one_lapse_biasinside_alpha_beta_phra_diagsigma_student4mu_C5_cholmuinit"
    files = filter(x -> occursin(match_str, x), files)
    res_dir = joinpath(base_path, files[1])
    beta_str = "model_var_betas.npz.npy"
    alpha_str = "model_var_alpha.npz.npy"
    sigma_str = "model_var_sigma.npz.npy"
    lapse_str = "model_var_lapse.npz.npy"
    data = Dict(
        "beta" => npzread(joinpath(res_dir, beta_str)),
        "lapse" => npzread(joinpath(res_dir, lapse_str)),
        "alpha" => npzread(joinpath(res_dir, alpha_str)),
        "sigma" => npzread(joinpath(res_dir, sigma_str))
    )
    return data
end


function conf_intervals_betas(samples)
    _, nday, ncol = size(samples)
    Q = fill(NaN, nday, ncol, 2)
    for day in 1:nday
        for col in 1:ncol
            Q[day, col, :] = quantile(samples[:, day, col], [0.025, 0.975])
        end
    end
    return Q
end

function proc_choice_weights(path)
    fips = [collect(13:16); collect(26:43)]
    choice_weights = Dict("mu" => Dict(), "95conf" => Dict(), "5conf" => Dict())
    lr_rates = Dict("mu" => Dict(), "95conf" => Dict(), "5conf" => Dict())
    lapse_rates = Dict("mu" => Dict(), "95conf" => Dict(), "5conf" => Dict())
    for fip in fips
        data = load_stan_res(path, fip)
        choice_weights["mu"][fip] = drop_dim(mean(data["beta"]; dims=1)) 
        qbeta = conf_intervals_betas(data["beta"]) 
        choice_weights["95conf"][fip] = qbeta[:, :, 2]
        choice_weights["5conf"][fip] = qbeta[:, :, 1]
        lr_rates["mu"][fip] = drop_dim(mean(data["alpha"]; dims=1))
        qlr = quantile(data["alpha"], [0.025, 0.975])
        lr_rates["95conf"][fip] = qlr[2]
        lr_rates["5conf"][fip] = qlr[1]
        lapse_rates["mu"][fip] = drop_dim(mean(data["lapse"]; dims=1))
        qlapse = hcat([quantile(data["lapse"][:, day], [0.025, 0.975]) for day in 1:size(data["lapse"], 2)]...)'
        lapse_rates["95conf"][fip] = qlapse[:, 2]
        lapse_rates["5conf"][fip] = qlapse[:, 1]
    end
    return choice_weights, lr_rates, lapse_rates
end



function choice_kernel(alpha, y)
    N = length(y)

    choice_map = Dict(1 => 1, 0 => -1)

    choice = [choice_map[y_i] for y_i in y]

    ck = zeros(N)

    for t in 1:N-1
        ck[t+1] = ck[t] + alpha * (choice[t] - ck[t])
    end

    return ck
end


function load_beh_data(fip, data_path)
    X_str = string("fip_", fip, "_choice_X_orig_rtcheck.npy")

    y_str = string("fip_", fip, "_choice_y_orig_rtcheck.npy")

    sesmap_str = string("fip_", fip, "_choice_sesmap_orig_rtcheck.npy")

    X = npzread(joinpath(data_path, X_str))

    y = npzread(joinpath(data_path, y_str))

    sesmap = npzread(joinpath(data_path, sesmap_str))

    bools = [sesmap .== j for j in 1:maximum(sesmap)]

    x_days = [X[b, :] for b in bools]

    y_days = [y[b] for b in bools] 

    return (
        X = x_days,
        y = y_days ,
        sesmap = sesmap
    )
end


mult = Dict(1 => -1, 2 => 1)
cohs = [-1, -0.25, -0.125, -0.0625, 0.0625, 0.125, 0.25, 1]

contra_ipsi_cols = Dict(
    13 => Dict("contra" => 2, "ipsi" => 1),
    14 => Dict("contra" => 2, "ipsi" => 1),
    15 => Dict("contra" => 2, "ipsi" => 1),
    16 => Dict("contra" => 2, "ipsi" => 1),
    26 => Dict("contra" => 1, "ipsi" => 2),
    27 => Dict("contra" => 1, "ipsi" => 2),
    28 => Dict("contra" => 1, "ipsi" => 2),
    29 => Dict("contra" => 1, "ipsi" => 2),
    30 => Dict("contra" => 2, "ipsi" => 1), 
    31 => Dict("contra" => 2, "ipsi" => 1),
    32 => Dict("contra" => 2, "ipsi" => 1),
    33 => Dict("contra" => 2, "ipsi" => 1), 
    34 => Dict("contra" => 2, "ipsi" => 1), 
    35 => Dict("contra" => 2, "ipsi" => 1),
    36 => Dict("contra" => 2, "ipsi" => 1),
    37 => Dict("contra" => 1, "ipsi" => 2), 
    38 => Dict("contra" => 1, "ipsi" => 2), 
    39 => Dict("contra" => 2, "ipsi" => 1),
    40 => Dict("contra" => 1, "ipsi" => 2),
    41 => Dict("contra" => 1, "ipsi" => 2),
    42 => Dict("contra" => 2, "ipsi" => 1),
    43 => Dict("contra" => 2, "ipsi" => 1),
    901 => Dict("contra" => 1, "ipsi" => 2),
    902 => Dict("contra" => 1, "ipsi" => 2), 
    903 => Dict("contra" => 2, "ipsi" => 1),
    904 => Dict("contra" => 2, "ipsi" => 1),
    905 => Dict("contra" => 2, "ipsi" => 1),
    906 => Dict("contra" => 1, "ipsi" => 2),
    907 => Dict("contra" => 1, "ipsi" => 2),
    908 => Dict("contra" => 2, "ipsi" => 1),
    910 => Dict("contra" => 2, "ipsi" => 1),
    911 => Dict("contra" => 1, "ipsi" => 2),
    912 => Dict("contra" => 1, "ipsi" => 2),
    913 => Dict("contra" => 1, "ipsi" => 2),
    914 => Dict("contra" => 2, "ipsi" => 1)
)

ipsi_contra_cols = Dict(
    13 => Dict("contra" => 1, "ipsi" => 2),
    14 => Dict("contra" => 1, "ipsi" => 2),
    15 => Dict("contra" => 1, "ipsi" => 2),
    16 => Dict("contra" => 1, "ipsi" => 2),
    26 => Dict("contra" => 2, "ipsi" => 1),
    27 => Dict("contra" => 2, "ipsi" => 1),
    28 => Dict("contra" => 2, "ipsi" => 1),
    29 => Dict("contra" => 2, "ipsi" => 1),
    30 => Dict("contra" => 1, "ipsi" => 2), 
    31 => Dict("contra" => 1, "ipsi" => 2),
    32 => Dict("contra" => 1, "ipsi" => 2),
    33 => Dict("contra" => 1, "ipsi" => 2), 
    34 => Dict("contra" => 1, "ipsi" => 2), 
    35 => Dict("contra" => 1, "ipsi" => 2),
    36 => Dict("contra" => 1, "ipsi" => 2),
    37 => Dict("contra" => 2, "ipsi" => 1), 
    38 => Dict("contra" => 2, "ipsi" => 1), 
    39 => Dict("contra" => 1, "ipsi" => 2),
    40 => Dict("contra" => 2, "ipsi" => 1),
    41 => Dict("contra" => 2, "ipsi" => 1),
    42 => Dict("contra" => 1, "ipsi" => 2),
    43 => Dict("contra" => 1, "ipsi" => 2),
    901 => Dict("contra" => 1, "ipsi" => 2),# 1 is right
    902 => Dict("contra" => 1, "ipsi" => 2), # 2 is left
    903 => Dict("contra" => 2, "ipsi" => 1),
    904 => Dict("contra" => 2, "ipsi" => 1),
    905 => Dict("contra" => 2, "ipsi" => 1),
    906 => Dict("contra" => 1, "ipsi" => 2),
    907 => Dict("contra" => 1, "ipsi" => 2),
    908 => Dict("contra" => 2, "ipsi" => 1),
    910 => Dict("contra" => 2, "ipsi" => 1),
    911 => Dict("contra" => 1, "ipsi" => 2),
    912 => Dict("contra" => 1, "ipsi" => 2),
    913 => Dict("contra" => 1, "ipsi" => 2),
    914 => Dict("contra" => 2, "ipsi" => 1)
)



reg_contra = Dict(
    "nacc" => ipsi_contra_cols,
    "dms" => contra_ipsi_cols,
    "dls" => ipsi_contra_cols,
)

contra_ipsi_fc = Dict(
    13 => Dict("contra" => 6, "ipsi" => 4),
    14 => Dict("contra" => 6, "ipsi" => 4),
    15 => Dict("contra" => 6, "ipsi" => 4),
    16 => Dict("contra" => 6, "ipsi" => 4),
    26 => Dict("contra" => 4, "ipsi" => 6),
    27 => Dict("contra" => 4, "ipsi" => 6),
    28 => Dict("contra" => 4, "ipsi" => 6),
    29 => Dict("contra" => 4, "ipsi" => 6),
    30 => Dict("contra" => 6, "ipsi" => 4),
    31 => Dict("contra" => 6, "ipsi" => 4),
    32 => Dict("contra" => 6, "ipsi" => 4),
    33 => Dict("contra" => 6, "ipsi" => 4),
    34 => Dict("contra" => 6, "ipsi" => 4),
    35 => Dict("contra" => 6, "ipsi" => 4),
    36 => Dict("contra" => 6, "ipsi" => 4),
    37 => Dict("contra" => 4, "ipsi" => 6),
    38 => Dict("contra" => 4, "ipsi" => 6),
)

contra_ipsi_fi = Dict(
    13 => Dict("contra" => 7, "ipsi" => 5),
    14 => Dict("contra" => 7, "ipsi" => 5),
    15 => Dict("contra" => 7, "ipsi" => 5),
    16 => Dict("contra" => 7, "ipsi" => 5),
    26 => Dict("contra" => 5, "ipsi" => 7),
    27 => Dict("contra" => 5, "ipsi" => 7),
    28 => Dict("contra" => 5, "ipsi" => 7),
    29 => Dict("contra" => 5, "ipsi" => 7),
    30 => Dict("contra" => 7, "ipsi" => 5),
    31 => Dict("contra" => 7, "ipsi" => 5),
    32 => Dict("contra" => 7, "ipsi" => 5),
    33 => Dict("contra" => 7, "ipsi" => 5),
    34 => Dict("contra" => 7, "ipsi" => 5),
    35 => Dict("contra" => 7, "ipsi" => 5),
    36 => Dict("contra" => 7, "ipsi" => 5),
    37 => Dict("contra" => 5, "ipsi" => 7),
    38 => Dict("contra" => 5, "ipsi" => 7),
)

sex = Dict(
    13 => 1,
    14 => 1,
    15 => 1,
    16 => 1,
    26 => 1,
    27 => 1,
    28 => 1,
    29 => 1,
    30 => 1,
    31 => 1,
    32 => 1,
    33 => 1,
    34 => 2,
    35 => 2,
    36 => 2,
    37 => 2,
    38 => 2,
    39 => 1,
    40 => 1,
    41 => 2,
    42 => 2,
    43 => 2    
)

bias_side = Dict(
    13 => 0,
    14 => 2,
    15 => 1,
    16 => 2,
    26 => 1,
    27 => 0,
    28 => 1,
    29 => 0,
    30 => 1,
    31 => 0,
    32 => 1,
    33 => 1,
    34 => 1,
    35 => 1,
    36 => 1,
    37 => 2,
    38 => 1,
    39 => 2,
    40 => 2,
    41 => 1,
    42 => 2,
    43 => 2,
)

optofip = [collect(901:908); collect(910:914)]
stim_groups = Dict(
    901 => "YFP",
    902 => "ChRmine",
    903 => "ChRmine",
    904 => "ChRmine",
    905 => "YFP",
    906 => "YFP",
    907 => "ChRmine",
    908 => "YFP",
    910 => "ChRmine",
    911 => "ChRmine",
    912 => "ChRmine",
    913 => "YFP",
    914 => "YFP"
)


stim_side = Dict(
    901 => Dict("contra" => 1, "ipsi" => 2),# 1 is right
    902 => Dict("contra" => 1, "ipsi" => 2), # 2 is left
    903 => Dict("contra" => 2, "ipsi" => 1),
    904 => Dict("contra" => 2, "ipsi" => 1),
    905 => Dict("contra" => 2, "ipsi" => 1),
    906 => Dict("contra" => 1, "ipsi" => 2),
    907 => Dict("contra" => 1, "ipsi" => 2),
    908 => Dict("contra" => 2, "ipsi" => 1),
    910 => Dict("contra" => 2, "ipsi" => 1),
    911 => Dict("contra" => 1, "ipsi" => 2),
    912 => Dict("contra" => 1, "ipsi" => 2),
    913 => Dict("contra" => 1, "ipsi" => 2),
    914 => Dict("contra" => 2, "ipsi" => 1)
)


mult_map(x) = x == 2 ? 1 : -1

mults_opto_contra = Dict(f => mult_map(stim_side[f]["contra"]) for f in [collect(901:908); collect(910:914)])

ns_map(x) = x == 1 ? 2 : 1

non_stim_side = Dict(f => ns_map(stim_side[f]["contra"]) for f in [collect(901:908); collect(910:914)] )

mults_opto_ipsi = Dict(f => mult_map(stim_side[f]["ipsi"]) for f in [collect(901:908); collect(910:914)] )


function bias(w, thresh)
    day = findfirst(abs.(w) .>= thresh)
    if isnothing(day)
        return length(w)
    end
    return day
end

fluo_colors_nacc = ["#012a4a", "#014f86", "#468faf", "#a9d6e5"]
fluo_colors_dms = ["#621b00", "#bc3908", "#ff9e00", "#ffcd7d"]
fluo_colors_dls = ["#1e441e", "#2a7221", "#31cb00", "#96e072"]

function choice_stats(data)
    left_choice_prop = sum(data.choice .== -1) / size(data, 1)

    no_choice_prop = sum(data.choice .== 0) / size(data, 1)

    right_choice_prop = sum(data.choice .== 1) / size(data, 1)

    rt_r = data.feedback_time[data.choice .== 1] .- data.cue_time[data.choice .== 1]

    rt_l = data.feedback_time[data.choice .== -1] .- data.cue_time[data.choice .== -1]

    avg_rt_r = mean(rt_r)

    avg_rt_l = mean(rt_l)

    sd_rt_r = std(rt_r)

    sd_rt_l = std(rt_l)

    persev_prop = sum(data.choice[1:end-1] .== data.choice[2:end]) / size(data, 1)

    acc_r = mean(data.rewarded[data.signed_contrast .> 0])

    acc_l = mean(data.rewarded[data.signed_contrast .< 0])
    return (
        
    acc = [mean(data.rewarded), acc_r, acc_l],

    stats = [
        right_choice_prop, avg_rt_r, sd_rt_r, 
        left_choice_prop, avg_rt_l, sd_rt_l, 
        persev_prop,
        no_choice_prop
        ]
    )
end

stim_map = Dict(1 => "sr", 2 => "sl")
fc_map = Dict(1 => "frc", 2 => "flc")
fi_map = Dict(1 => "fri", 2 => "fli")
actc_map = Dict(1 => "arc", 2 => "alc")
acti_map = Dict(1 => "ari", 2 => "ali")

fips = [collect(13:16); collect(26:43)]



function avg_first_day_kernel(neu, event_dict, reg_contra)
    contra = fill(NaN, 3, 22, 50, 4);
    ipsi = fill(NaN, 3, 22, 50, 4);

    for (reg_k, reg) in enumerate(["nacc", "dms", "dls"])
        for (fip_j, fip) in enumerate(fips)
            contra[reg_k, fip_j, :, :] = neu[fip].full_res[reg][event_dict[reg_contra[reg][fip]["contra"]]][1, :, :]
            ipsi[reg_k, fip_j, :, :] = neu[fip].full_res[reg][event_dict[reg_contra[reg][fip]["ipsi"]]][1, :, :]
        end
    end

    avg_contra = reshape(mean(contra; dims=2), 3, 50, 4)

    avg_ipsi = reshape(mean(ipsi; dims=2), 3, 50, 4)

    std_contra = reshape(std(contra; dims=2), 3, 50, 4) ./ sqrt(22)

    std_ipsi = reshape(std(ipsi; dims=2), 3, 50, 4) ./ sqrt(22)

    return (
        avg_contra = avg_contra,
        avg_ipsi = avg_ipsi,
        sd_contra = std_contra,
        sd_ipsi = std_ipsi
    )
end

function avg_first_day_norm(neu, event_dict, reg_contra)
    contra = fill(NaN, 3, 22, 4);
    ipsi = fill(NaN, 3, 22, 4);

    for (reg_k, reg) in enumerate(["nacc", "dms", "dls"])
        for (fip_j, fip) in enumerate(fips)
            contra[reg_k, fip_j, :] = vec(mapslices(norm, neu[fip].full_res[reg][event_dict[reg_contra[reg][fip]["contra"]]][1, :, :]; dims=1))
            ipsi[reg_k, fip_j, :] = vec(mapslices(norm, neu[fip].full_res[reg][event_dict[reg_contra[reg][fip]["ipsi"]]][1, :, :]; dims=1))
        end
    end

    avg_contra = reshape(mean(contra; dims=2), 3, 4)

    avg_ipsi = reshape(mean(ipsi; dims=2), 3, 4)

    std_contra = reshape(std(contra; dims=2), 3, 4) ./ sqrt(22)

    std_ipsi = reshape(std(ipsi; dims=2), 3, 4) ./ sqrt(22)

    return (
        avg_contra = avg_contra,
        avg_ipsi = avg_ipsi,
        sd_contra = std_contra,
        sd_ipsi = std_ipsi
    )
end


function conmatrix(cons)
    ucon = [0.0625, 0.125, 0.25, 1]
    n_trial = length(cons)
    Z = zeros(Bool, n_trial, 4)
    for (j, c) in enumerate(ucon)
        ii_j = cons .== c
        Z[ii_j, j] .= true
    end
    return Z
end

sder(X) = vec(nanstd(X, 1)) ./ sqrt(size(X, 1))

function check_pval(pval)
    if pval < 1e-12
        pstr = "888"
    elseif pval < 1e-11
        pstr = "< 1e-11"
    elseif pval < 1e-10
        pstr = "< 1e-10"
    elseif pval < 1e-9
        pstr = "< 1e-9"
    elseif pval < 1e-8
        pstr = "< 1e-8"
    elseif pval < 1e-7
        pstr = "< 1e-7"
    elseif pval < 1e-6
        pstr = "< 1e-6"
    elseif pval < 0.00001
        pstr = "< 1e-5"
    elseif pval < 0.0001
        pstr = "< 1e-4"
    elseif pval < 0.001
        pstr = "< 1e-3"
    else
        pstr = string(pval)
    end
    return pstr
end

function pval_stars(pval)
    if pval < 0.0001
        pstr = "***"
    elseif pval < 0.001
        pstr = "**"
    elseif pval < 0.05
        pstr = "*"
    else 
        pstr = ""
    end
    return pstr
end

