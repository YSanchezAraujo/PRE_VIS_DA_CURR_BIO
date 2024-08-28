include("utility.jl");

using NPZ;
using DataFrames;
using StatsBase;
using JSON;
using Interpolations;

function get_session_paths(base_path, fip; stable_sub_path_str = "alf")
    path_info = []
    data_path = joinpath(base_path, "fip_$fip")
    outside_paths = [joinpath(data_path, x) for x in readdir(data_path) if isdir(joinpath(data_path, x))]
    neural_missing = 0
    
    for op in outside_paths
        numbered_dirs = [x for x in readdir(op) if !occursin(".DS", x)]
        session_number = npzread(joinpath(op, numbered_dirs[1], "training_day.npy"))[1]

        if isfile(joinpath(op, numbered_dirs[1], "no_neural.flag"))
            neural_missing = true
        end
  
        multiple_tries = length(numbered_dirs) > 1 ? true : false

        if neural_missing == 1
            qc_dict = Dict("NAcc" => NaN, "DMS" => NaN, "DLS" => NaN)
        else
            qc_dict = JSON.parsefile(joinpath(op, numbered_dirs[1], stable_sub_path_str, "FP_QC.json"))
        end

        push!(path_info, 
                (
                    joinpath(op, numbered_dirs[1], stable_sub_path_str),
                    session_number, 
                    multiple_tries,
                    neural_missing,
                    qc_dict["NAcc"],
                    qc_dict["DMS"],
                    qc_dict["DLS"]
                )
        )
        
        neural_missing = 0
    end

    return DataFrame(path_info, [:path, :session, :multi, :neural_missing, :QC_NAcc, :QC_DMS, :QC_DLS])
end

function get_neural_data(path)
    fluo_names = [
        "_ibl_trials.NAcc.npy",
        "_ibl_trials.DMS.npy",
        "_ibl_trials.DLS.npy",
        "_ibl_fluo.times_corrected.npy"
    ]

    neural_data = hcat(npzread.(joinpath.(path, fluo_names))...)

    with_nans = DataFrame(
        NAcc = neural_data[:, 1],
        DMS = neural_data[:, 2],
        DLS = neural_data[:, 3],
        times = neural_data[:, 4]
    )

    bools_not_nans = drop_dim(mapslices(any, .!isnan.(neural_data); dims=2))
    without_nans = DataFrame(neural_data[bools_not_nans, :], [:NAcc, :DMS, :DLS, :times])
    return with_nans, without_nans
end

function get_trial_behavior(path)
    beh_names = [
        "_ibl_trials.goCueTrigger_times.npy",
        "_ibl_trials.firstMovement_times.npy",
        "_ibl_trials.feedback_times.npy",
        "_ibl_trials.feedbackType.npy",
        "_ibl_trials.choice.npy",
        "_ibl_trials.contrastLeft.npy",
        "_ibl_trials.contrastRight.npy"
    ]

    behavior_data = hcat(npzread.(joinpath.(path, beh_names))...)

    return DataFrame(
        stim_time = behavior_data[:, 1],
        act_time = behavior_data[:, 2],
        reward_time = behavior_data[:, 3],
        reward_type = behavior_data[:, 4],
        choice = behavior_data[:, 5],
        contrast_left = behavior_data[:, 6],
        contrast_right = behavior_data[:, 7]
    )
end


function find_broken_trials(fluo_time, trial_time)
    return isnothing.([findfirst(fluo_time .>= trial_time[k]) for k in 1:length(trial_time)])
end

function working_trial_bools(fluo_time, stim_time, act_time, reward_time)
    return (
        find_broken_trials(fluo_time, stim_time) + 
        find_broken_trials(fluo_time, act_time) +
        find_broken_trials(fluo_time, reward_time)
    ) .== 0
end

function get_event_match_idx(fluo_time, trial_time)
    return [findfirst(fluo_time .>= trial_time[k]) for k in 1:length(trial_time)]
end

function event_indices(fluo_time, stim_time, act_time, reward_time)
    return (
        get_event_match_idx(fluo_time, stim_time),
        get_event_match_idx(fluo_time, act_time),
        get_event_match_idx(fluo_time, reward_time)
    )
end

function get_wheel_data(path)
    wheel_pos = npzread(joinpath(path, "_ibl_wheel.position.npy"))
    wheel_pos = mod.(wheel_pos, 6.283185307179586)
    wheel_pos = wheel_pos .- wheel_pos[1]
    wheel_times = npzread(joinpath(path, "_ibl_wheel.timestamps.npy"))
    return [wheel_times wheel_pos]
end


scipy_signal = pyimport("scipy.signal")
function vel_gauss_conv(pos, rate; smooth_size=0.03)
    std_samps = Int64(round(smooth_size * (1 / rate)))
    N = std_samps * 6
    gauss_std = (N - 1) / 6
    win = scipy_signal.gaussian(N, gauss_std)
    win = win ./ sum(win)
    vel = [0; scipy_signal.convolve(diff(pos), win, mode="same")] .* (1 / rate)
    acc = [0; scipy_signal.convolve(diff(vel), win, mode="same")] .* (1 / rate)
    return vel, acc
end

function resample_wheel_data(neural_times, wheel_time, wheel_position; pad=0.0)
    start_time = minimum(neural_times) - pad
    end_time = maximum(neural_times) + pad    
    dtn = mean(diff(neural_times))
    new_time_points = range(start_time, length=length(neural_times), step=dtn)    
    interp_time = LinearInterpolation(wheel_time, wheel_time, extrapolation_bc = Line())
    interp_position = LinearInterpolation(wheel_time, wheel_position, extrapolation_bc = Line())
    resampled_wheel_time = interp_time(new_time_points)
    resampled_wheel_position = interp_position(new_time_points)
    start_time = minimum(neural_times) - pad
    end_time = maximum(neural_times) + pad
    truncated_indices = (new_time_points .>= start_time) .& (new_time_points .<= end_time)
    truncated_wheel_time = resampled_wheel_time[truncated_indices]
    truncated_wheel_position = resampled_wheel_position[truncated_indices]
    return truncated_wheel_time, truncated_wheel_position
end

"""
the main function for loading the behavioral and fluorescence data,
it applies some preprocessing steps
""";
function mouse_session_data(base_path, mouse_id, day; target_hz=50)
    path_info = get_session_paths(base_path, mouse_id)
    session_idx = findfirst(path_info.session .== day)

    if session_idx == nothing
        println("No session found for day $(day)")
        return nothing
    end

    path = path_info.path[session_idx]

    """
    intial loading of both data modalities
    """
    _, neural_data = get_neural_data(path)
    behavior_data = get_trial_behavior(path)
    behavior_data.rt = behavior_data.reward_time .- behavior_data.stim_time
    """
    resample neural data, this may be redundant but for consistency
    we resample the neural data to a 1/target_hz
    """
    neural_data = DataFrame(
        NAcc = zscore(resample_data(neural_data.times, neural_data.NAcc; rate = 1/target_hz)),
        DMS = zscore(resample_data(neural_data.times, neural_data.DMS; rate = 1/target_hz)),
        DLS = zscore(resample_data(neural_data.times, neural_data.DLS; rate = 1/target_hz)),
        times = resample_data(neural_data.times, neural_data.times; rate = 1/target_hz)
    )
    """
    now we are going to try and align the neural and behavioral data
    if there are trials before neural data was collected, discard them
    """
    first_beh_trial = findfirst(neural_data.times[1] .< behavior_data.stim_time)
    behavior_data = behavior_data[first_beh_trial:end, :]
    """
    removing any trials where the animal didnt make a choice, or some error occured
    in the time stamp saving. these type of trials luckily are not common, even at the start of training
    """;
    keep_trial_bools = working_trial_bools(neural_data.times, behavior_data.stim_time, behavior_data.act_time, behavior_data.reward_time)
    behavior_data = behavior_data[keep_trial_bools, :]
    """
    get the indices w.r.t to the fluo data for each event across all trials
    """
    stim_idx, act_idx, reward_idx = event_indices(neural_data.times, behavior_data.stim_time, behavior_data.act_time, behavior_data.reward_time)
    """
    loading in the wheel data and resampling it
    """;
    wheel_data = get_wheel_data(path)
    wheel_time, wheel_pos = resample_wheel_data(neural_data.times, wheel_data[:, 1], wheel_data[:, 2])
    wheel_vel, wheel_acel = vel_gauss_conv(wheel_pos, 1 / target_hz)
    wheel_data = DataFrame([wheel_time zscore(wheel_pos) zscore(wheel_vel) zscore(wheel_acel)], [:times, :pos, :vel, :acel])

    return (
        neural = neural_data,
        behavior = behavior_data,
        wheel = wheel_data,
        stim_idx = stim_idx,
        act_idx = act_idx,
        reward_idx = reward_idx,
        path = path,
        session_number = path_info.session[session_idx],
        mult_tries = path_info.multi[session_idx]
    )
end

"""
this function is the equivalant of the above function but for day 0 
"""
function mouse_session0_data(base_path, mouse_id; target_hz=50)
    path_info = get_session_paths(base_path, mouse_id)
    session_idx = findfirst(path_info.session .== 0)
    path = path_info.path[session_idx]
    _, neural_data = get_neural_data(path)
    behavior_data = get_trial_behavior(path)

    neural_data = DataFrame(
        NAcc = zscore(resample_data(neural_data.times, neural_data.NAcc; rate = 1/target_hz)),
        DMS = zscore(resample_data(neural_data.times, neural_data.DMS; rate = 1/target_hz)),
        DLS = zscore(resample_data(neural_data.times, neural_data.DLS; rate = 1/target_hz)),
        times = resample_data(neural_data.times, neural_data.times; rate = 1/target_hz)
    )

    # discard negative time
    first_pos_idx_neural = findfirst(neural_data.times .> 0)
    neural_data = neural_data[first_pos_idx_neural:end, :]

    first_beh_trial = findfirst(neural_data.times[1] .< behavior_data.stim_time)
    behavior_data = behavior_data[first_beh_trial:end, :]

    stim_idx, act_idx, reward_idx = event_indices(neural_data.times, behavior_data.stim_time, behavior_data.act_time, behavior_data.reward_time)

    use_stim_idx = .!isnothing.(stim_idx)

    return (
        neural = neural_data,
        behavior = behavior_data[use_stim_idx, :],
        stim_idx = stim_idx[.!isnothing.(stim_idx)],
        act_idx = act_idx[.!isnothing.(act_idx)],
        reward_idx = reward_idx[.!isnothing.(reward_idx)],
        path = path,
        session_number = path_info.session[session_idx],
        mult_tries = path_info.multi[session_idx]
    )
end
