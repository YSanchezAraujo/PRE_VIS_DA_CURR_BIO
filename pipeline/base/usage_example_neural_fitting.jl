using PyPlot;
include(joinpath(@__DIR__, "encoding_model.jl"))
include(joinpath(@__DIR__, "preprocess.jl"))
include(joinpath(@__DIR__, "design_matrix.jl"))
include(joinpath(@__DIR__, "constants.jl"))


# set parameters
day = 10 # session number to load
nfunc = 16 # number of basis functions
window = 50 # number of time bins for temporal kernels, spans 1 second
mouse_id = 27
n_stim = 4 # 4 stimulus levels: 6.25%, 12.5%, 25%, 100%
n_sets = 10 # this corresponds to the number of events in the design matrix, see event names below
base_path = "/jukebox/witten/ONE/alyx.internationalbrainlab.org/wittenlab/Subjects" # whereever you have downloaded the data
wheel_path = "/jukebox/witten/Alex/Data/Subjects/"

# load data and create design matrices
data = mouse_session_data(base_path, mouse_id, day);
desmat_items = make_design_matrix_items(data, window, nfunc);
#full_desmat = [design_matrix(desmat_items) data.wheel.pos data.wheel.vel data.wheel.acel];
full_desmat = design_matrix(desmat_items);
wheel_for_desmat = wheel_desmat_features(data, window);


# this design matrix is truncated to only behavioral events 
trunc_desmat_items = truncate_design_matrix([full_desmat wheel_for_desmat], data, window);
#trunc_desmat_items = truncate_design_matrix(full_desmat, data, window);

X = vcat(trunc_desmat_items.X...);
Y = vcat(trunc_desmat_items.Y...);

# fit the encoding model for each region
model_fits = [bayes_ridge(X, Y[:, reg])  for reg in 1:3];

plot_reg = 2
model_fit = model_fits[plot_reg]
reg_labels = ["NAcc", "DMS", "DLS"]

# extract the weights in the standard basis
W = desmat_items.basis * reshape(model_fit.w[2:end-6], (nfunc, n_stim * n_sets))
S = desmat_items.basis * reshape(sqrt.(diag(model_fit.covar)[2:end-6]), (nfunc, n_stim * n_sets))


# put the estimated model estimates in a less error prone data structure
kernels = weights_by_event(W, n_stim, event_names)
errors = weights_by_event(S, n_stim, event_names)

# for some selected event type, plot the average response
xt = collect(0:49) ./ 49
plot_event = event_names[1]
plt.figure()
for i in 1:4
    plt.plot(xt, kernels[plot_event][:, i])
    plt.fill_between(
        xt, 
        kernels[plot_event][:, i] .- errors[plot_event][:, i],
        kernels[plot_event][:, i] .+ errors[plot_event][:, i],
        alpha=0.5, label="_nolegend_"
        )
end
plt.xlabel("seconds")
plt.ylabel(latexstring("\\boldsymbol{K}_{\\text{stimulus}}"))
plt.title("region: $(reg_labels[plot_reg]) estimated kernel ")
plt.legend(["6.25%", "12.5%", "25%", "100%"])

# use estimated weights to predict neural data
Y_hat = hcat([X*model_fits[reg].w for reg in 1:3]...)

# compute how good of a fit the model is
var_expl = [rsquared(Y[:, reg], Y_hat[:, reg]) for reg in 1:3]

inds = collect(zip(trunc_desmat_items.stim_idx, trunc_desmat_items.stim_idx .+ (window - 1)))[1:end-1]

# look at time locked predictions
avg_stim = average_signal(Y[:, plot_reg], inds, window)
avg_pred = average_signal(Y_hat[:, plot_reg], inds, window)

plt.figure()
plt.plot(xt, avg_stim.mean, label="true", lw=3)
plt.plot(xt, avg_pred.mean, label="predicted", linestyle="--", lw=3)
plt.xlabel("seconds")
plt.ylabel(latexstring("\\frac{\\Delta f}{f}"))
plt.title("time locked reward response across event types")
plt.legend()
