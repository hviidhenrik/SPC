from typing import Optional, Union, List

import matplotlib.axes._subplots
import numpy as np
import pandas as pd
import sklearn.decomposition
import sklearn.preprocessing
from matplotlib import pyplot as plt


def flatten_list(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]


def get_df_with_sample_id(df: pd.DataFrame, n_sample_size: int = 1):
    assert n_sample_size > 0, "Sample size must be a positive integer > 0"
    df_copy = df.copy()
    n_obs = len(df)
    n_subgroups = int(np.ceil(n_obs / n_sample_size))
    sample = np.repeat([i for i in range(1, n_subgroups + 1)], n_sample_size)
    df_copy["sample_id"] = list(sample)[:n_obs]
    # rearrange columns to have sample_id as the first column, if it isn't already
    cols = df_copy.columns.values.tolist()
    if not cols[0] == "sample_id":
        cols = cols[-1:] + cols[:-1]
    return df_copy[cols]


def multiply_matrices(*arrays: np.array):
    """
    Helper function to multiply an arbitrary number of matrices with appropriate dimensions wrt each other.

    :param arrays: the matrices, given as numpy arrays, to be multiplied together
    :return: the matrix product
    """
    product = arrays[0]
    for array in arrays[1:]:
        product = np.matmul(product, array)
    return product


def standardize_and_PCA(df, n_components: Optional[int] = None):
    scaler = sklearn.preprocessing.StandardScaler().fit(df)
    df_transformed = scaler.transform(df)
    pca = sklearn.decomposition.PCA(n_components=n_components).fit(df_transformed)
    df_transformed = pca.transform(df_transformed)
    return df_transformed, pca, scaler


class ControlChartPlotMixin:

    def __init__(self):
        self.UCL = None
        self.LCL = None
        self.center_line = None
        self.input_name = None
        self.stat_name = None

    @staticmethod
    def _plot_scalar_or_array(x: Union[float, np.ndarray, List[float]],
                              ax: matplotlib.axes._subplots.Axes):
        if isinstance(x, (np.ndarray, list)):
            ax.plot(x, color="red", linestyle="dashed")  # , label="Control limits")
        else:
            ax.axhline(x, color="red", linestyle="dashed", label="Control limits")

    def _plot_single_phase(self, df, y_limit_offsets=(0.95, 1.05)):
        fig, ax = plt.subplots(1, 1)
        ax.plot(df[self.stat_name], linestyle="-", marker="o", color="black")
        legend_labels = [self.stat_name]
        if self.center_line is not None:
            ax.axhline(self.center_line, color="blue", alpha=0.7)
            legend_labels.append("Center line")
        if self.UCL is not None:
            self._plot_scalar_or_array(self.UCL, ax)
            legend_labels.append("Control limit")
        if self.LCL is not None:
            self._plot_scalar_or_array(self.LCL, ax)

        plt.legend(legend_labels, ncol=len(legend_labels))
        y_limits = ax.get_ylim()
        ax.set_ylim(y_limits[0] * y_limit_offsets[0], y_limits[1] * y_limit_offsets[1])
        return fig

    def _plot_two_phases(self, df_phase1: pd.DataFrame, df_phase2: pd.DataFrame, y_limit_offsets=(0.95, 1.05)):
        fig, ax = plt.subplots(1, 1)
        df = pd.concat([df_phase1, df_phase2])
        df["phase"] = 1
        df["phase"].iloc[len(df_phase1):] = 2
        df = df.reset_index()

        legend_labels = ["Phase 1", "Phase 2"]
        plt.plot(df[self.stat_name][df["phase"] == 1],
                 linestyle="-", marker="o", color="green")
        plt.plot(df[self.stat_name][df["phase"] == 2],
                 linestyle="-", marker="o", color="orange")
        if self.center_line is not None:
            ax.axhline(self.center_line, color="blue", alpha=0.7)
            legend_labels.append("Center line")
        if self.UCL is not None:
            self._plot_scalar_or_array(self.UCL, ax)
            legend_labels.append("Control limit")
        if self.LCL is not None:
            self._plot_scalar_or_array(self.LCL, ax)

        plt.legend(legend_labels, ncol=len(legend_labels))
        y_limits = ax.get_ylim()
        ax.set_ylim(y_limits[0] * y_limit_offsets[0], y_limits[1] * y_limit_offsets[1])
        return fig
