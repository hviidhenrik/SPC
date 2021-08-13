"""
Unit tests go here
"""
import pytest
from pandas._testing import assert_frame_equal
from phdspc.core import *


def test_xbarchart_fit_correct_values_sample_size_2():
    df_phase1 = pd.DataFrame({"x1": [1, 2, 4, 5, 8, 9, 5]})
    fitted = XBarChart(m_sample_size=2).fit(df_phase1=df_phase1)

    expected_center_line = np.mean([1.5, 4.5, 8.5, 5])
    constant_A2_m2 = 1.88
    expected_Rbar = np.mean([1, 1, 1, 0])
    expected_UCL = expected_center_line + constant_A2_m2 * expected_Rbar
    expected_LCL = expected_center_line - constant_A2_m2 * expected_Rbar

    assert expected_center_line == fitted.center_line
    assert expected_Rbar == fitted.Rbar
    assert expected_LCL == fitted.LCL
    assert expected_UCL == fitted.UCL


def test_xbarchart_fit_correct_values_sample_size_4():
    df_phase1 = pd.DataFrame({"x1": [1, 2, 4, 1, 1, 3, 2]})
    fitted = XBarChart(m_sample_size=4).fit(df_phase1=df_phase1)

    expected_center_line = np.mean([2, 2])
    constant_A2_m4 = 0.729
    expected_Rbar = np.mean([3, 2])
    expected_UCL = expected_center_line + constant_A2_m4 * expected_Rbar
    expected_LCL = expected_center_line - constant_A2_m4 * expected_Rbar

    assert expected_center_line == fitted.center_line
    assert expected_Rbar == fitted.Rbar
    assert expected_LCL == fitted.LCL
    assert expected_UCL == fitted.UCL


# TODO implement tests for R chart:
def test_Rchart_fit_correct_values_sample_size_2():
    pass


def test_Rchart_fit_correct_values_sample_size_4():
    pass

