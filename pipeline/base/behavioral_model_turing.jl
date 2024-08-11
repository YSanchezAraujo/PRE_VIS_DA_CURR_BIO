using Turing;
using LogExpFunctions;
using Distributions;
using StatsBase;
using LinearAlgebra;
using PDMats;

function side_choice_kern(choice, lr, N, dt)
    choice_kernel = zeros(dt, N)

    for t in 1:N-1
        choice_kernel[t+1] = choice_kernel[t] + lr * (choice[t] - choice_kernel[t])
    end

    return choice_kernel
end

function ses_side_choice_kern(
    choice, 
    lr, 
    N, 
    S, 
    ST, 
    SE, 
    NS,
    dt
    )

    all_choice_k = zeros(dt, N)

    for s in 1:S
        all_choice_k[ST[s]:SE[s]] .= side_choice_kern(choice[ST[s]:SE[s]], lr, NS[s], dt)
    end

    return all_choice_k
end

@model function hierarchical_logistic_regression(X, y, choice, ST, SE, NS, sesmap)
    n_col = size(X, 2)
    n_samp = length(y)
    n_ses = length(NS)

    sigma ~ filldist(truncated(Normal(0, 10), lower=0), n_col+1)
    eta ~ truncated(Normal(0, 10), lower=0)
    L ~ LKJCholesky(n_col+1, eta)
    Cov_L = Diagonal(sigma) * L.L
    Cov = PDMat(Cholesky(Cov_L + eps() * I))
    nu ~ Gamma(2.0, 5.0)
    alpha ~ truncated(Normal(0, 5), lower=0.0)
    mu_init ~ filldist(Normal(0, 5), n_col+1)
    betas = Vector{Vector{Float64}}(undef, n_ses)
    betas[1] ~ MvTDist(nu, mu_init, Cov)
    lr_pre_transform ~ Normal(0, 1)
    lr = logistic(lr_pre_transform)

    choice_kern = zeros(typeof(nu), n_samp)
    for s in 1:n_ses
        choice_kern[ST[s]:SE[s]] .= side_choice_kern(choice[ST[s]:SE[s]], lr, NS[s], typeof(nu))
    end
    
    for ses in 2:n_ses
        betas[ses] ~ MvTDist(nu, betas[ses-1], Cov)
    end

    coh_alpha = tanh(alpha)
    means = vec(
        sum(
            [X[:, 1] tanh.(X[:, 2:3], alpha) ./ coh_alpha choice_kern] .* hcat(betas[sesmap]...)';
                dims=2)
        )
    
    y .~ BernoulliLogit.(means)
end

# example usage
# model = hierarchical_logistic_regression(
#     X, Float64.(y), choice, session_start, session_end, beh_data.trials, sesmap
#     );

# chain = sample(model, NUTS(), 5_00);

# n_day = length(beh_data.day)
# bias = [mean(chain[Symbol("betas[$day][1]")]) for day in 1:n_day]
# left = [mean(chain[Symbol("betas[$day][2]")]) for day in 1:n_day]
# right = [mean(chain[Symbol("betas[$day][3]")]) for day in 1:n_day]
# chist = [mean(chain[Symbol("betas[$day][4]")]) for day in 1:n_day]