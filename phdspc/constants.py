# TODO fill these out for more sample sizes.

"""
The constants for control charts below are based on the table of Appendix VI of
Statistical Quality Control, 2013, 7th Edition by Douglas C. Montgomery.
"""
_A2 = {"2": 1.88, "3": 1.023, "4": 0.729, "5": 0.577, "6": 0.483, "7": 0.419}
_D3 = {"2": 0, "3": 0, "4": 0, "5": 0, "6": 0, "7": 0.076}
_D4 = {"2": 3.267, "3": 2.574, "4": 2.282, "5": 2.114, "6": 2.004, "7": 1.924}


def get_A2(m_sample_size: int = 2):
    return _A2[str(m_sample_size)]


def get_D3(m_sample_size: int = 2):
    return _D3[str(m_sample_size)]


def get_D4(m_sample_size: int = 2):
    return _D4[str(m_sample_size)]

