include(joinpath(@__DIR__, "utility.jl"));

function get_beh_dirs(base_dir)
    return [x for x in readdir(base_dir) if occursin("mouse_", x)]
end

function load_npy_file(path)
    data = npzread(path)
    variable_name = split(split(path, "/")[end], "_")[1]
    return data, variable_name
end

function load_beh_result(data_path, mouse_id)
    result_dirs = get_beh_dirs(data_path)
    mouse_dir = filter(x -> occursin(string(mouse_id), x), result_dirs)[1]
    full_path = joinpath(data_path, mouse_dir)
    
    var_dict = Dict()
    var_avg_dict = Dict()
    var_5_quant_dict = Dict()
    var_95_quant_dict = Dict()

    for file in readdir(full_path)
        var_data, name = load_npy_file(joinpath(full_path, file))
        avg_var = drop_dim(mean(var_data; dims=1))
        
        if length(size(var_data)) == 1
            var_5_quant = quantile(var_data, 0.025)
            var_95_quant = quantile(var_data, 0.975)
        else
            var_5_quant = drop_dim(mapslices(x -> quantile(x, 0.025), var_data; dims=1))
            var_95_quant = drop_dim(mapslices(x -> quantile(x, 0.975), var_data; dims=1))
        end

        setindex!(var_dict, var_data, name)
        setindex!(var_avg_dict, avg_var, name)

        setindex!(var_5_quant_dict, var_5_quant, name)
        setindex!(var_95_quant_dict, var_95_quant, name)
    end

    return (
        avg = var_avg_dict,
        samples = var_dict,
        quant_5 = var_5_quant_dict,
        quant_95 = var_95_quant_dict
    )
end

saved_results_path = "/jukebox/witten/yoel/saved_results";

mouse_ids = [collect(13:16); collect(26:43)];

betas = Dict()

# TODO: save the rest of the variables
for mouse in mouse_ids
    res = load_beh_result(saved_results_path, mouse)
    setindex!(betas, MouseBehavior(res.avg["betas"], res.quant_5["betas"], res.quant_95["betas"]), mouse)
end

save(
    joinpath(saved_results_path, "choice_weights.jld2"),
    "results",
    betas
)
