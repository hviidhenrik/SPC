"""
Unit tests go here
"""
import pytest
from pandas._testing import assert_frame_equal
from phdspc.helpers import *


def test_flatten_list():
    assert flatten_list([[1, 1], [2, 2], [3, 3]]) == [1, 1, 2, 2, 3, 3]
    assert flatten_list([[], []]) == []
    assert flatten_list([[]]) == []


def test_get_df_with_sample_id():
    df_test_input = pd.DataFrame({"x1": [1, 2, 2, 3, 3, 4, 5]})
    df_expected1 = pd.DataFrame({"sample_id": [1, 2, 3, 4, 5, 6, 7],
                                 "x1": [1, 2, 2, 3, 3, 4, 5]})
    df_expected2 = pd.DataFrame({"sample_id": [1, 1, 2, 2, 3, 3, 4],
                                 "x1": [1, 2, 2, 3, 3, 4, 5]})
    df_expected3 = pd.DataFrame({"sample_id": [1, 1, 1, 1, 1, 1, 1],
                                 "x1": [1, 2, 2, 3, 3, 4, 5]})
    df_output1 = get_df_with_sample_id(df_test_input, n_sample_size=1)
    df_output2 = get_df_with_sample_id(df_test_input, n_sample_size=2)
    df_output3 = get_df_with_sample_id(df_test_input, n_sample_size=7)

    assert_frame_equal(df_output1, df_expected1, check_dtype=False)
    assert_frame_equal(df_output2, df_expected2, check_dtype=False)
    assert_frame_equal(df_output3, df_expected3, check_dtype=False)
    with pytest.raises(Exception):
        get_df_with_sample_id(df_test_input, n_sample_size=0)