# TODO fill these out for more sample sizes.

"""
The constants for control charts below are based on the table of Appendix VI of
Statistical Quality Control, 2013, 7th Edition by Douglas C. Montgomery.
"""
_A2 = {
    "2": 1.88,
    "3": 1.023,
    "4": 0.729,
    "5": 0.577,
    "6": 0.483,
    "7": 0.419,
    "8": 0.373,
    "9": 0.337,
    "10": 0.308,
    "11": 0.285,
    "12": 0.266,
    "13": 0.249,
}
_A3 = {
    "2": 2.659,
    "3": 1.954,
    "4": 1.628,
    "5": 1.427,
    "6": 1.287,
    "7": 1.182,
    "8": 1.099,
    "9": 1.032,
    "10": 0.957,
    "11": 0.927,
    "12": 0.886,
    "13": 0.85,
}
_B3 = {
    "2": 0,
    "3": 0,
    "4": 0,
    "5": 0,
    "6": 0.03,
    "7": 0.118,
    "8": 0.185,
    "9": 0.239,
    "10": 0.284,
    "11": 0.321,
    "12": 0.354,
    "13": 0.382,
}
_B4 = {
    "2": 3.267,
    "3": 2.568,
    "4": 2.266,
    "5": 2.089,
    "6": 1.970,
    "7": 1.882,
    "8": 1.815,
    "9": 1.761,
    "10": 1.716,
    "11": 1.679,
    "12": 1.646,
    "13": 1.618,
}
_D3 = {
    "2": 0,
    "3": 0,
    "4": 0,
    "5": 0,
    "6": 0,
    "7": 0.076,
    "8": 0.136,
    "9": 0.184,
    "10": 0.223,
    "11": 0.256,
    "12": 0.283,
    "13": 0.307,
}
_D4 = {
    "2": 3.267,
    "3": 2.574,
    "4": 2.282,
    "5": 2.114,
    "6": 2.004,
    "7": 1.924,
    "8": 1.864,
    "9": 1.816,
    "10": 1.777,
    "11": 1.744,
    "12": 1.717,
    "13": 1.693,
}

_A_dict = {"2": _A2, "3": _A3}
_B_dict = {"3": _B3, "4": _B4}
_D_dict = {"3": _D3, "4": _D4}


def get_A_constant(A_number: int, n_sample_size: int = 2):
    return _A_dict[str(A_number)][str(n_sample_size)]


def get_B_constant(B_number: int, n_sample_size: int = 2):
    return _B_dict[str(B_number)][str(n_sample_size)]


def get_D_constant(D_number: int, n_sample_size: int = 2):
    return _D_dict[str(D_number)][str(n_sample_size)]
