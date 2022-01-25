"""
pytest fixtures go here
"""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def dataframe_for_PCAModel_phase1():
    np.random.seed(1234)
    df_phase1 = pd.DataFrame(
        {
            "x1": np.random.normal(loc=1, size=500),
            "x2": np.random.normal(loc=5, size=500),
            "x3": np.random.normal(loc=10, size=500),
        }
    )
    return df_phase1
