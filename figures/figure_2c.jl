include(joinpath(@__DIR__, "../base/preprocess.jl"));
include(joinpath(@__DIR__, "../base/constants.jl"));
include(joinpath(@__DIR__, "../base/design_matrix.jl"));
include(joinpath(@__DIR__, "../base/correlation_functions.jl"));
include(joinpath(@__DIR__, "../base/utility.jl"));
include(joinpath(@__DIR__, "../base/figboilerplate.jl"));
include(joinpath(@__DIR__, "../base/encoding_model.jl"));

# set parameters
day = 4 # session number to load
nfunc = 16 # number of basis functions
window = 50 # number of time bins for temporal kernels, spans 1 second
mouse_id = 27
n_stim = 4 # 4 stimulus levels: 6.25%, 12.5%, 25%, 100%
n_sets = 10 # this corresponds to the number of events in the design matrix, see event names below
base_path = "/jukebox/witten/ONE/alyx.internationalbrainlab.org/wittenlab/Subjects" # whereever you have downloaded the data

# load data and create design matrices
data = mouse_session_data(base_path, mouse_id, day);
desmat_items = make_design_matrix_items(data, window, nfunc);
#full_desmat = [design_matrix(desmat_items) data.wheel.pos data.wheel.vel data.wheel.acel];
full_desmat = design_matrix(desmat_items);
wheel_for_desmat = wheel_desmat_features(data, window);


# this design matrix is truncated to only behavioral events 
trunc_desmat_items = truncate_design_matrix([full_desmat wheel_for_desmat], data, window);

X = vcat(trunc_desmat_items.X...);
Y = vcat(trunc_desmat_items.Y...);

# fit the encoding model for each region
model_fits = [bayes_ridge(X, Y[:, reg])  for reg in 1:3];
Y_hat = hcat([X*model_fits[reg].w for reg in 1:3]...)
var_expl = [rsquared(Y[:, reg], Y_hat[:, reg]) for reg in 1:3]

trs = [
    trunc_desmat_items.stim_idx[j]:trunc_desmat_items.reward_idx[j]+(window - 1)
    for j in [ 150, 200, 230, 235, 240]
]

#NACC
fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(10, 2))
for i in 1:5
    ax[i].plot(Y_hat[trs[i], 1], lw=3, color="tab:blue")
    ax[i].plot(Y[trs[i], 1], linestyle="--", color="black", lw=3)
end
for i in 1:5
    ax[i].spines["left"].set_visible(false)
    ax[i].spines["bottom"].set_visible(false)
    ax[i].set_xticks([])
    ax[i].set_yticks([])
    ax[i].set_ylim(-1, 7)
end
r2ex = round(var_expl[1]; digits=2)
ax[3].set_title(latexstring("\$R^2: $r2ex\$"), fontsize=23)
plt.savefig("figure_2c_nacc_preds.pdf", bbox_inches="tight", transparent=true)

#DMS
fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(10, 2))
for i in 1:5
    ax[i].plot(Y_hat[trs[i], 2], lw=3, color="tab:orange")
    ax[i].plot(Y[trs[i], 2], linestyle="--", color="black", lw=3)
end
for i in 1:5
    ax[i].spines["left"].set_visible(false)
    ax[i].spines["bottom"].set_visible(false)
    ax[i].set_xticks([])
    ax[i].set_yticks([])
    ax[i].set_ylim(-1, 7)
end
r2ex = round(var_expl[2]; digits=2)
ax[3].set_title(latexstring("\$R^2: $r2ex\$"), fontsize=23)
plt.savefig("figure_2c_dms_preds.pdf", bbox_inches="tight", transparent=true)

#DLS
fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(10, 2))
for i in 1:5
    ax[i].plot(Y_hat[trs[i], 3], lw=3, color="tab:green")
    ax[i].plot(Y[trs[i], 3], linestyle="--", color="black", lw=3)
end
for i in 1:5
    ax[i].spines["left"].set_visible(false)
    ax[i].spines["bottom"].set_visible(false)
    ax[i].set_xticks([])
    ax[i].set_yticks([])
    ax[i].set_ylim(-1, 7)
end
r2ex = round(var_expl[3]; digits=2)
ax[3].set_title(latexstring("\$R^2: $r2ex\$"), fontsize=23)
plt.savefig("figure_2c_dls_preds.pdf", bbox_inches="tight", transparent=true)


