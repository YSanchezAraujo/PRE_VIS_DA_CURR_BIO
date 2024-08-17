include(joinpath(@__DIR__, "utility.jl"));
include(joinpath(@__DIR__, "preprocess.jl"));

function get_beh_dirs(base_dir)
    return [x for x in readdir(base_dir) if occursin("mouseid_", x) && !occursin(".npy", x) && !occursin(".jld2", x)]
end

function load_npy_file(path)
    data = npzread(path)
    variable_name = split(split(path, "/")[end], "_")[1]
    return data, variable_name
end

function load_beh_result(data_path, mouse_id)
    result_dirs = get_beh_dirs(data_path)
    mouse_dir = filter(x -> occursin("mouseid_$(mouse_id)", x), result_dirs)[1]
    full_path = joinpath(data_path, mouse_dir)
    
    var_dict = Dict()
    var_avg_dict = Dict()
    var_max_quant_dict = Dict()
    var_min_quant_dict = Dict()

    for file in readdir(full_path)
        var_data, name = load_npy_file(joinpath(full_path, file))
        avg_var = drop_dim(mean(var_data; dims=1))
        
        if length(size(var_data)) == 1
            var_min_quant = quantile(var_data, 0.025)
            var_max_quant = quantile(var_data, 0.975)
        else
            var_min_quant = drop_dim(mapslices(x -> quantile(x, 0.025), var_data; dims=1))
            var_max_quant = drop_dim(mapslices(x -> quantile(x, 0.975), var_data; dims=1))
        end

        setindex!(var_dict, var_data, name)
        setindex!(var_avg_dict, avg_var, name)

        setindex!(var_min_quant_dict, var_min_quant, name)
        setindex!(var_max_quant_dict, var_max_quant, name)
    end

    return (
        avg = var_avg_dict,
        samples = var_dict,
        quant_min = var_min_quant_dict,
        quant_max = var_max_quant_dict
    )
end

saved_results_path = "/jukebox/witten/yoel/saved_results";
data_path = "/jukebox/witten/ONE/alyx.internationalbrainlab.org/wittenlab/Subjects"

mouse_ids = [collect(13:16); collect(26:43)];

betas = Dict()

# TODO: save the rest of the variables
for mouse in mouse_ids
    session_paths = get_session_paths(data_path, mouse)
    fx(x) = x == 0 ? false : true
    non_pretrain_session = fx.(session_paths.session)
    day_labels = session_paths.session[non_pretrain_session]
    
    res = load_beh_result(saved_results_path, mouse)
    temp_avg = fill(NaN, 20, size(res.avg["betas"], 2))
    temp_quant_min = fill(NaN, 20, size(res.quant_min["betas"], 2))
    temp_quant_max = fill(NaN, 20, size(res.quant_max["betas"], 2))

    temp_avg[day_labels, :] = res.avg["betas"]
    temp_quant_min[day_labels, :] = res.quant_min["betas"]
    temp_quant_max[day_labels, :] = res.quant_max["betas"]

    setindex!(betas, MouseBehavior(temp_avg, temp_quant_min, temp_quant_max), mouse)
end

save(
    joinpath(saved_results_path, "choice_weights.jld2"),
    "results",
    betas
)