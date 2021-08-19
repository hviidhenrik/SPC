"""
Unit tests go here
"""
import pytest
from pandas._testing import assert_frame_equal

from phdspc.core import *


def sd(values):
    return np.std(values, ddof=1)


def test_xbarchart_fit_correct_values_sample_size_2_and_range():
    df_phase1 = pd.DataFrame({"x1": [1, 2, 4, 5, 8, 9]})
    fitted = XBarChart(n_sample_size=2).fit(df_phase1=df_phase1)

    expected_center_line = np.mean([1.5, 4.5, 8.5])
    constant_A2 = 1.88
    expected_Rbar = np.mean([1, 1, 1])
    expected_LCL = expected_center_line - constant_A2 * expected_Rbar
    expected_UCL = expected_center_line + constant_A2 * expected_Rbar

    assert expected_center_line == fitted.center_line
    assert expected_Rbar == fitted.variability_Rbar_or_sbar
    assert expected_LCL == fitted.LCL
    assert expected_UCL == fitted.UCL


def test_xbarchart_fit_correct_values_sample_size_4_and_range():
    df_phase1 = pd.DataFrame({"x1": [1, 2, 4, 1, 1, 3, 2]})
    fitted = XBarChart(n_sample_size=4, variability_estimator="range").fit(df_phase1=df_phase1)

    expected_center_line = np.mean([2, 2])
    constant_A2 = 0.729
    expected_Rbar = np.mean([3, 2])
    expected_LCL = expected_center_line - constant_A2 * expected_Rbar
    expected_UCL = expected_center_line + constant_A2 * expected_Rbar

    assert expected_center_line == fitted.center_line
    assert expected_Rbar == fitted.variability_Rbar_or_sbar
    assert expected_LCL == fitted.LCL
    assert expected_UCL == fitted.UCL


def test_get_df_with_sample_means_and_variability_sample_size_2_and_range():
    df_test_input = pd.DataFrame({"sample_id": [1, 1, 2, 2, 3, 3, 4],
                                  "x1": [1, 2, 2, 3, 3, 4, 5]})
    df_expected = pd.DataFrame({"sample_mean": [1.5, 2.5, 3.5, 5],
                                "sample_variability": [1, 1, 1, 0]},
                               index=[1, 2, 3, 4])
    chart = XBarChart(n_sample_size=2, variability_estimator="range").fit(df_test_input)
    df_expected.index.name = "sample_id"
    df_output = chart._get_df_with_sample_means_and_variability(df_test_input, grouping_column="sample_id")
    assert_frame_equal(df_output, df_expected, check_dtype=False)


def test_get_df_with_sample_means_and_variability_sample_size_12_and_std():
    values = np.linspace(1, 24, 24)
    df_test_input = pd.DataFrame({"sample_id": [1] * 12 + [2] * 12,
                                  "x1": values
                                  })
    df_expected = pd.DataFrame({"sample_mean": [6.5, 18.5],
                                "sample_variability": [np.std(values[0:12], ddof=1), np.std(values[12:24], ddof=1)]
                                },
                               index=[1, 2])
    chart = XBarChart(n_sample_size=12, variability_estimator="std").fit(df_test_input)
    df_expected.index.name = "sample_id"
    df_output = chart._get_df_with_sample_means_and_variability(df_test_input, grouping_column="sample_id")
    assert_frame_equal(df_output, df_expected, check_dtype=False)


def test_group_samples_and_compute_stats():
    df_test_input = pd.DataFrame({"x1": [1, 2, 4, 3, 2, 3, 2]})
    fitted = XBarChart(n_sample_size=2).fit(df_phase1=df_test_input)

    df_output = fitted._group_samples_and_compute_stats(df_test_input)
    df_expected = pd.DataFrame({"sample_mean": [1.5, 3.5, 2.5, 2],
                                "sample_variability": [1, 1, 1, 0]},
                               index=[1, 2, 3, 4])
    df_expected.index.name = "sample_id"
    assert_frame_equal(df_output, df_expected, check_dtype=False)


def test_collect_results_df_correct_output_dataframe():
    df_test_input = pd.DataFrame({"x1": [1, 2, 4, 3, 2, 3, 2]})
    fitted = XBarChart(n_sample_size=2, variability_estimator="range").fit(df_phase1=df_test_input)

    df_output = fitted._collect_results_df(fitted.df_phase1_stats)
    df_expected = pd.DataFrame({fitted.stat_name: [1.5, 3.5, 2.5],
                                "LCL": [0.62] * 3,
                                "UCL": [4.38] * 3,
                                "outside_CL": [False] * 3,
                                },
                               index=[1, 2, 3])
    df_expected.index.name = "sample_id"
    assert_frame_equal(df_output, df_expected, check_dtype=False)


def test_collect_results_df_correct_bool_for_outside_CL():
    df_test_input = pd.DataFrame({"x1": [1, 2, 4, 3, 2, 3, 2, 20]})
    fitted = XBarChart(n_sample_size=2, variability_estimator="range").fit(df_phase1=df_test_input)
    fitted.LCL = 1.5
    fitted.UCL = 3.5
    df_output = fitted._collect_results_df(fitted.df_phase1_stats)

    df_expected = pd.DataFrame({fitted.stat_name: [1.5, 3.5, 2.5, 11.0],
                                "LCL": [1.5] * 4,
                                "UCL": [3.5] * 4,
                                "outside_CL": [True, True, False, True],
                                },
                               index=[1, 2, 3, 4])
    df_expected.index.name = "sample_id"
    assert_frame_equal(df_output, df_expected, check_dtype=False)


def test_RChart_fit_correct_values_sample_size_2_and_range():
    df_phase1 = pd.DataFrame({"x1": [1, 3, 6, 7, 1, 3, 2]})
    fitted = RChart(n_sample_size=2, variability_estimator="range").fit(df_phase1=df_phase1)

    constant_D3 = 0
    constant_D4 = 3.267
    expected_Rbar = np.mean([2, 1, 2])
    expected_LCL = constant_D3 * expected_Rbar
    expected_UCL = constant_D4 * expected_Rbar

    assert expected_Rbar == fitted.variability_Rbar_or_sbar
    assert expected_LCL == fitted.LCL
    assert expected_UCL == fitted.UCL


def test_RChart_fit_correct_values_sample_size_4_and_std():
    df_phase1 = pd.DataFrame({"x1": [1, 3, 6, 7, 1, 3, 2, 4, 5]})
    fitted = RChart(n_sample_size=4, variability_estimator="std").fit(df_phase1=df_phase1)

    constant_B3 = 0
    constant_B4 = 2.266
    expected_sbar = np.mean([sd([1, 3, 6, 7]), sd([1, 3, 2, 4])])
    expected_LCL = constant_B3 * expected_sbar
    expected_UCL = constant_B4 * expected_sbar

    assert expected_sbar == fitted.variability_Rbar_or_sbar
    assert expected_LCL == fitted.LCL
    assert expected_UCL == fitted.UCL


def test_RChart_fit_correct_values_sample_size_4_and_range():
    df_phase1 = pd.DataFrame({"x1": [1, 2, 4, 1, 2, 5, 12]})
    fitted = RChart(n_sample_size=4, variability_estimator="range").fit(df_phase1=df_phase1)

    constant_D3 = 0
    constant_D4 = 2.282
    expected_Rbar = np.mean([3, 10])
    expected_LCL = constant_D3 * expected_Rbar
    expected_UCL = constant_D4 * expected_Rbar

    assert expected_Rbar == fitted.variability_Rbar_or_sbar
    assert expected_LCL == fitted.LCL
    assert expected_UCL == fitted.UCL


@pytest.mark.parametrize("test_variability_estimator, test_sample_size, expected_output",
                         [("range", 2, "range"), ("Range", 30, "range"), ("std", 2, "std"), ("Std", 30, "std"),
                          ("std", 2, "std"), ("Auto", 10, "range"), ("auto", 11, "std")])
def test_determine_varability_estimator_correct_output(test_variability_estimator, test_sample_size, expected_output):
    xbarchart = XBarChart(n_sample_size=test_sample_size, variability_estimator=test_variability_estimator)
    assert expected_output == xbarchart._determine_variability_estimator(test_variability_estimator)


@pytest.mark.parametrize("test_bad_variability_estimator_string",
                         ["r", "a", "s", "mkaæomkm", "foo", "variance", None, -1, True, False])
def test_determine_varability_estimator_bad_input_fails(test_bad_variability_estimator_string):
    with pytest.raises(Exception):
        XBarChart(variability_estimator=test_bad_variability_estimator_string)


def test_remove_samples_with_only_one_observation():
    chart = XBarChart(n_sample_size=2)
    chart.input_name = "x1"
    df_test_input = pd.DataFrame({"sample_id": [1, 1, 2, 2, 3],
                                  "x1": [1] * 5})
    df_output = chart._remove_samples_with_only_one_observation(df_test_input)
    df_expected = pd.DataFrame({"sample_id": [1, 1, 2, 2],
                                "x1": [1] * 4})
    assert_frame_equal(df_expected, df_output, check_dtype=False)


def test_schart_correct_init():
    chart = SChart(n_sample_size=2)
    assert chart.variability_estimator == "std"


def test_schart_fit_correct_values():
    df_phase1 = pd.DataFrame({"x1": [1, 3, 6, 7, 1, 3, 2, 4, 5]})
    fitted = SChart(n_sample_size=4).fit(df_phase1=df_phase1)

    constant_B3 = 0
    constant_B4 = 2.266
    expected_sbar = np.mean([sd([1, 3, 6, 7]), sd([1, 3, 2, 4])])
    expected_LCL = constant_B3 * expected_sbar
    expected_UCL = constant_B4 * expected_sbar

    assert expected_sbar == fitted.variability_Rbar_or_sbar
    assert expected_LCL == fitted.LCL
    assert expected_UCL == fitted.UCL


def test_MEWMA_fit_correct_values():
    """
    This test follows the results obtained and data used in the numerical example provided in the original
    article from 1992 by Lowry et al.: "A Multivariate Exponentially Weighted Moving Average Control Chart".
    """
    df_phase2 = pd.DataFrame({"x1": [-1.19, 0.12, -1.69, 0.3, 0.89, 0.82, -0.3, 0.63, 1.56, 1.46],
                              "x2": [0.59, 0.9, 0.4, 0.46, -0.75, 0.98, 2.28, 1.75, 1.58, 3.05]
                              })
    chart = MEWMAChart(lambda_=0.1, sigma=np.array([[1, 0.5], [0.5, 1]])).fit(df_phase2)
    T2_output = np.array(chart.df_phase2_stats["T2"]).round(2)
    Z1_output = np.array(chart.df_phase2_stats["Z1"]).round(2)
    Z2_output = np.array(chart.df_phase2_stats["Z2"]).round(2)

    Z1_expected = np.array([-0.12, -0.1, -0.25, -0.2, -0.09, 0., -0.03, 0.04, 0.19, 0.32])
    Z2_expected = np.array([0.06, 0.14, 0.17, 0.2, 0.1, 0.19, 0.4, 0.53, 0.64, 0.88])
    T2_expected = np.array([3.29, 3.18, 7.37, 5.26, 1.09, 1.28, 5.66, 8.32, 9.64, 17.21])

    assert (T2_output == T2_expected).all()
    assert (Z1_output == Z1_expected).all()
    assert (Z2_output == Z2_expected).all()


def test_MEWMA_compute_delta():
    """ Using the example correlation matrix in chapter 11.4, p. 527 and the shifted mu vector of [1, 1, 1, 1, 1, 1].
    The value of delta should then be 1.86.
    """
    chart = MEWMAChart()
    chart.is_fitted = True
    chart.sigma = np.array([[1, 0.7, 0.9, 0.3, 0.2, 0.3],
                            [0.7, 1, 0.8, 0.1, 0.4, 0.2],
                            [0.9, 0.8, 1, 0.1, 0.2, 0.1],
                            [0.3, 0.1, 0.1, 1, 0.2, 0.1],
                            [0.2, 0.4, 0.2, 0.2, 1, 0.1],
                            [0.3, 0.2, 0.1, 0.1, 0.1, 1]])
    delta_output = np.round(chart.compute_delta(np.array([1] * 6)), 2)
    delta_expected = 1.86
    assert delta_output == delta_expected
