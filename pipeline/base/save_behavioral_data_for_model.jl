using JLD2;
include("/Users/ysa/Desktop/pipeline/base/preprocess.jl");


function mouse_behavioral_data_all_days(base_path, mouse_id)
    session_paths = get_session_paths(base_path, mouse_id)
    num_sessions = maximum(session_paths.session)
    fx(x) = x == 0 ? false : true
    non_pretrain_session = fx.(session_paths.session)
    trials_per_day = Int64[]
    beh_datasets, choices = [], []
    day_labels = session_paths.session[non_pretrain_session]
    
    for path in session_paths.path[non_pretrain_session]
        behavior_data = get_trial_behavior(path)
        contrasts = [behavior_data.contrast_left behavior_data.contrast_right]
        contrasts[isnan.(contrasts)] .= 0.0
        push!(trials_per_day, size(contrasts, 1))
        push!(beh_datasets, add_intercept(contrasts))
        push!(choices, behavior_data.choice)
    end

    return (
        data = beh_datasets,
        day = day_labels,
        trials = trials_per_day,
        choice = choices
    )

end

base_path = "/Users/ysa/Desktop/Subjects"
save_path = "/Users/ysa/Desktop/pipeline/saved_results"
fy(c) = c == 1 ? 1 : 0


for mouse_id in [collect(13:16); collect(26:43)]
    beh_data = mouse_behavioral_data_all_days(base_path, mouse_id)
    X = vcat(beh_data.data...)
    choice = vcat(beh_data.choice...)
    y = fy.(choice)
    session_end = cumsum(beh_data.trials)
    session_start = [1; session_end[1:end-1] .+ 1]
    sesmap = Int64.(vcat((ones.(beh_data.trials) .* collect(1:length(beh_data.day))...)))

    npzwrite(
        joinpath(save_path, "behavior_data_mouseid_$(mouse_id).npy"),
        Dict(
            "X" => X,
            "y" => y,
            "session_start" => session_start,
            "session_end" => session_end,
            "trial_per_session" => beh_data.trials,
            "day" => beh_data.day,
            "sesmap" => sesmap,
            "choice" => choice
        )
    )
end
