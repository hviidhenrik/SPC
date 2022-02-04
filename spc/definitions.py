univariate_phase1_color = "green"
univariate_phase2_color = "black"
multivariate_phase1_color = "green"
multivariate_phase2_color = "black"
outlier_marker_color = "red"
UCL_color = "red"
LCL_color = "red"
single_phase_line_color = "black"


def get_outside_CL_marker_size(N_samples: int):
    return 15 if N_samples > 500 else 25


def get_line_width(N_samples: int):
    return 1 if N_samples > 500 else 1.5
