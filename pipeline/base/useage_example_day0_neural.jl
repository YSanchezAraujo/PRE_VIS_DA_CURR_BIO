using PyPlot;
include(joinpath(@__DIR__, "encoding_model.jl"))
include(joinpath(@__DIR__, "preprocess.jl"))
include(joinpath(@__DIR__, "design_matrix.jl"))
include(joinpath(@__DIR__, "constants.jl"))


# set parameters
day = 0 # session number to load
nfunc = 16 # number of basis functions
window = 50 # number of time bins for temporal kernels, spans 1 second
mouse_id = 26
n_stim = 4 # 4 stimulus levels: 6.25%, 12.5%, 25%, 100%
n_sets = 2 # this corresponds to the number of events in the design matrix, see event names below
base_path = "/jukebox/witten/ONE/alyx.internationalbrainlab.org/wittenlab/Subjects" # whereever you have downloaded the data

# load data and create design matrices
data = mouse_session0_data(base_path, mouse_id);
desmat_items = make_design_matrix_items_day0(data, window, nfunc)
full_desmat = design_matrix_day0(desmat_items);

# this design matrix is truncated to only behavioral events 
trunc_desmat_items = truncate_design_matrix_day0(full_desmat, data, window);
X = vcat(trunc_desmat_items.X...);
Y = vcat(trunc_desmat_items.Y...);

model_fits = [bayes_ridge(X, zscore(Y[:, reg]))  for reg in 1:3];

plot_reg = 2
model_fit = model_fits[plot_reg]
reg_labels = ["NAcc", "DMS", "DLS"]

# extract the weights in the standard basis
W = desmat_items.basis * reshape(model_fit.w[2:end], (nfunc, n_stim * n_sets))
S = desmat_items.basis * reshape(sqrt.(diag(model_fit.covar)[2:end]), (nfunc, n_stim * n_sets))

plot_reg = 3
model_fit = model_fits[plot_reg]
reg_labels = ["NAcc", "DMS", "DLS"]

# extract the weights in the standard basis
W = desmat_items.basis * reshape(model_fit.w[2:end], (nfunc, n_stim * n_sets))
S = desmat_items.basis * reshape(sqrt.(diag(model_fit.covar)[2:end]), (nfunc, n_stim * n_sets))

kernels = weights_by_event(W, n_stim, event_names[1:n_sets])
errors = weights_by_event(S, n_stim, event_names[1:n_sets])