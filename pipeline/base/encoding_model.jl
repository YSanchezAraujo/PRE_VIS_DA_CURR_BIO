using LinearAlgebra;
using Distributions;
using StatsBase;

sse(x, y) = sum((x .- y).^2)

function bayes_ridge(X::AbstractMatrix, y::AbstractVector; tol::Float64=1e-2, max_iter::Int=50)
    n_samp, n_col = size(X)
    mu = zeros(n_col)
    I_n = I(n_col)

    XX = Matrix(X'X)
    Xy = X'y

    w_ols = (XX + I_n * 5) \ Xy
    sig2y = sse(y, X*w_ols)
    alpha = 5 / sig2y

    S = (1 / sig2y * XX .+ I_n * alpha) \ I_n
    mu = 1 / sig2y * S * Xy
    gammas = 1 .- alpha .* diag(S)
    alpha = (n_col - alpha * tr(S)) ./ sum(mu.^2)
    sig2y = sse(y, X*mu) / (n_samp - sum(gammas))

    params = [alpha, sig2y]
    for iter in 2:max_iter
        S = (1 / sig2y * XX .+ I_n * alpha) \ I_n
        mu = 1 / sig2y * S * Xy
        gammas = 1 .- alpha .* diag(S)
        alpha = (n_col - alpha * tr(S)) ./ sum(mu.^2)
        sig2y = sse(y, X*mu)  / (n_samp - sum(gammas))

        if (iter > 1) && (norm(params .- [alpha, sig2y]) < tol)
            println(string("converged in $iter", " iterations"))
            break
        end

        params = [alpha, sig2y]
    end

    return (
        sig2y = sig2y,
        w = mu, 
        alpha = alpha,
        covar = S, 
    )
end