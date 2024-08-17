functions {
    vector side_choice_kern(array[] int choice, real alpha, int N) {
        vector[N] choice_kernel = rep_vector(0, N);


        for (t in 1:N-1) {
            choice_kernel[t+1] = choice_kernel[t] + alpha * (choice[t] - choice_kernel[t]);
        }

        return choice_kernel;

    }

    vector ses_side_choice_kern(array[] int choice,
                             real alpha,
                             int N,
                             int S,
                             array[] int ST,
                             array[] int SE,
                             array[] int NS) {

        vector[N] all_choice_k;

        for (s in 1:S) {

            all_choice_k[ST[s]:SE[s]] = side_choice_kern(choice[ST[s]:SE[s]], alpha, NS[s]);

        }

        return all_choice_k;
    }

    real softplus(real x) {
        return log(1 + exp(x));
    }
}

data {
    int<lower=1> N; // number of samples

    int<lower=1> P; // number of columns

    int<lower=1> S; // number of sessions

    matrix[N, P] X;

    array[S] int NS;

    array[S] int ST;

    array[S] int SE;

    array[N] int<lower=0, upper=1> y;

    array[N] int<lower=-1, upper=1> choice;
    
    array[N] int<lower=1, upper=S> sesmap;

}

parameters {
    cholesky_factor_corr[P + 1] L;

    vector<lower=0>[P+1] sigma;

    array[S] vector[P + 1] betas;

    real<lower=0>alpha_pr;

    real lam_coh_alpha;

    real<lower=0.1>eta;

    vector[P+1] mu_init;

    real<lower=1>nu;
}

transformed parameters {

    real alpha = Phi_approx(alpha_pr);

    real coh_alpha = softplus(lam_coh_alpha);

    real tanh_cohalpha = tanh(coh_alpha);

    vector[N] choice_kern = ses_side_choice_kern(choice, alpha, N, S, ST, SE, NS);

    matrix[P+1, P+1] L_Sigma = diag_pre_multiply(sigma, L);

    vector[N] means;

    for (k in 1:S) {
        means[ST[k]:SE[k]] = ( 
            X[ST[k]:SE[k], 1] * betas[k][1] + 
            tanh(coh_alpha * X[ST[k]:SE[k], 2:3]) / tanh_cohalpha * betas[k][2:3] + 
            (choice_kern[ST[k]:SE[k]] * betas[k][P+1]) 
        );
    }
}

model {
    alpha_pr ~ normal(0, 1);

    lam_coh_alpha ~ normal(-2, 0.5);

    eta ~ normal(0, 10);

    L ~ lkj_corr_cholesky(eta);

    sigma ~ normal(0, 1.);

    nu ~ gamma(2, 0.2);

    mu_init ~ student_t(nu, 0, 5);

    betas[1] ~ multi_student_t_cholesky(nu, mu_init, L_Sigma);

    for (k in 2:S) {
        betas[k] ~ multi_student_t_cholesky(nu, betas[k-1], L_Sigma);
    }

   y ~ bernoulli_logit(means);
}
