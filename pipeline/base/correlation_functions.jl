"""
DMS CONTRA
""";
function contra_weights_dms_conmod(kernel_norm, behavior, mouseid)
    neu = (
        kernel_norm[mouseid]["DMS"][dms_contra_map[mouseid]][:, 4] .-
        kernel_norm[mouseid]["DMS"][dms_contra_map[mouseid]][:, 1]
    )

    beh = behavior[mouseid].avg[:, beh_dms_contra_map[mouseid]] * beh_multiplier[mouseid]

    return neu, beh
end

function contra_correlation_conmod_dms(kernel_norm, behavior, mouseid)
    return robust_nanlm(contra_weights_dms_conmod(kernel_norm, behavior, mouseid)...).cor
end

function avg_contra_correlation_conmod_dms(kernel_norm, behavior, mouse_ids)
    corr_set = [contra_correlation_conmod_dms(kernel_norm, behavior_weight, f) for f in mouse_ids]
    return (
        corrs = corr_set,
        avg_cor = mean(corr_set)
    )
end

"""
DMS IPSI
"""
function ipsi_weights_dms_conmod(kernel_norm, behavior, mouseid)
    neu = (
        kernel_norm[mouseid]["DMS"][dms_ipsi_map[mouseid]][:, 4] .-
        kernel_norm[mouseid]["DMS"][dms_ipsi_map[mouseid]][:, 1]
    )

    beh = behavior[mouseid].avg[:, beh_dms_ipsi_map[mouseid]] * (-1*beh_multiplier[mouseid])

    return neu, beh
end

function ipsi_correlation_conmod_dms(kernel_norm, behavior, mouseid)
    return robust_nanlm(ipsi_weights_dms_conmod(kernel_norm, behavior, mouseid)...).cor
end

function avg_ipsi_correlation_conmod_dms(kernel_norm, behavior, mouse_ids)
    corr_set = [ipsi_correlation_conmod_dms(kernel_norm, behavior_weight, f) for f in mouse_ids]
    return (
        corrs = corr_set,
        avg_cor = mean(corr_set)
    )
end

"""
DLS CONTRA
""";
function contra_weights_dls_conmod(kernel_norm, behavior, mouseid)
    neu = (
        kernel_norm[mouseid]["DLS"][dms_ipsi_map[mouseid]][:, 4] .-
        kernel_norm[mouseid]["DLS"][dms_ipsi_map[mouseid]][:, 1]
    )

    beh = behavior[mouseid].avg[:, beh_dms_ipsi_map[mouseid]] * (-1 * beh_multiplier[mouseid])

    return neu, beh
end

function contra_correlation_conmod_dls(kernel_norm, behavior, mouseid)
    return robust_nanlm(contra_weights_dls_conmod(kernel_norm, behavior, mouseid)...).cor
end

function avg_contra_correlation_conmod_dls(kernel_norm, behavior, mouse_ids)
    corr_set = [contra_correlation_conmod_dls(kernel_norm, behavior_weight, f) for f in mouse_ids]
    return (
        corrs = corr_set,
        avg_cor = mean(corr_set)
    )
end

"""
DLS IPSI
"""
function ipsi_weights_dls_conmod(kernel_norm, behavior, mouseid)
    neu = (
        kernel_norm[mouseid]["DLS"][dms_contra_map[mouseid]][:, 4] .-
        kernel_norm[mouseid]["DLS"][dms_contra_map[mouseid]][:, 1]
    )

    beh = behavior[mouseid].avg[:, beh_dms_contra_map[mouseid]] * beh_multiplier[mouseid]

    return neu, beh
end

function ipsi_correlation_conmod_dls(kernel_norm, behavior, mouseid)
    return robust_nanlm(ipsi_weights_dls_conmod(kernel_norm, behavior, mouseid)...).cor
end

function avg_ipsi_correlation_conmod_dls(kernel_norm, behavior, mouse_ids)
    corr_set = [ipsi_correlation_conmod_dls(kernel_norm, behavior_weight, f) for f in mouse_ids]
    return (
        corrs = corr_set,
        avg_cor = mean(corr_set)
    )
end



"""
NAcc CONTRA
""";
function contra_weights_nacc_conmod(kernel_norm, behavior, mouseid)
    neu = (
        kernel_norm[mouseid]["NAcc"][dms_ipsi_map[mouseid]][:, 4] .-
        kernel_norm[mouseid]["NAcc"][dms_ipsi_map[mouseid]][:, 1]
    )

    beh = behavior[mouseid].avg[:, beh_dms_ipsi_map[mouseid]] * (-1 * beh_multiplier[mouseid])

    return neu, beh
end

function contra_correlation_conmod_nacc(kernel_norm, behavior, mouseid)
    return robust_nanlm(contra_weights_nacc_conmod(kernel_norm, behavior, mouseid)...).cor
end

function avg_contra_correlation_conmod_nacc(kernel_norm, behavior, mouse_ids)
    corr_set = [contra_correlation_conmod_nacc(kernel_norm, behavior_weight, f) for f in mouse_ids]
    return (
        corrs = corr_set,
        avg_cor = mean(corr_set)
    )
end

"""
NAcc IPSI
"""
function ipsi_weights_nacc_conmod(kernel_norm, behavior, mouseid)
    neu = (
        kernel_norm[mouseid]["NAcc"][dms_contra_map[mouseid]][:, 4] .-
        kernel_norm[mouseid]["NAcc"][dms_contra_map[mouseid]][:, 1]
    )

    beh = behavior[mouseid].avg[:, beh_dms_contra_map[mouseid]] * beh_multiplier[mouseid]

    return neu, beh
end

function ipsi_correlation_conmod_nacc(kernel_norm, behavior, mouseid)
    return robust_nanlm(ipsi_weights_nacc_conmod(kernel_norm, behavior, mouseid)...).cor
end

function avg_ipsi_correlation_conmod_nacc(kernel_norm, behavior, mouse_ids)
    corr_set = [ipsi_correlation_conmod_nacc(kernel_norm, behavior_weight, f) for f in mouse_ids]
    return (
        corrs = corr_set,
        avg_cor = mean(corr_set)
    )
end

