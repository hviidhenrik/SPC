phase1_color = "black"
phase2_color = "black"
UCL_color = "red"
LCL_color = "red"
single_phase_line_color = "black"


def get_outside_CL_marker_size(N_samples: int):
    return 15 if N_samples > 500 else 25


def get_line_width(N_samples: int):
    return 1 if N_samples > 500 else 1.5
