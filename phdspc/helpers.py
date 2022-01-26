from typing import List, Optional, Tuple, Union

import matplotlib.axes._subplots
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.decomposition
import sklearn.preprocessing
from statsmodels.tsa.stattools import acf, pacf

from phdspc.definitions import *


def vprint(verbose: Union[bool, int], str_to_print: str, **kwargs):
    if verbose:
        print(str_to_print, **kwargs)


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


def multiply_matrices(*arrays: np.ndarray) -> np.float64:
    """
    Helper function to multiply an arbitrary number of matrices with appropriate dimensions wrt each other.

    :param arrays: the matrices, given as numpy arrays, to be multiplied together
    :return: the matrix product
    """
    product = arrays[0]
    for array in arrays[1:]:
        product = np.matmul(product, array)
    return product


def standardize_and_PCA(df: pd.DataFrame, n_components: Optional[int] = None):
    """
    Function to standardize the data to zero mean and unit variance and subsequently apply PCA.

    :param df: the dataframe of observations to apply transforms to
    :param n_components: optional number of principal components to retain, defaults to all
    :return: the standardized and PCA transformed dataframe, and the sklearn PCA and StandardScaler objects
    """
    scaler = sklearn.preprocessing.StandardScaler().fit(df)
    scaler.scale_ = np.std(df, ddof=1, axis=0).values  # use unbiased estimator instead of biased
    df_transformed = scaler.transform(df)
    pca = sklearn.decomposition.PCA(n_components=n_components).fit(df_transformed)
    df_transformed = pca.transform(df_transformed)
    return df_transformed, pca, scaler


def apply_standardize_and_PCA(
    df: pd.DataFrame,
    scaler: sklearn.preprocessing._data.StandardScaler,
    pca: sklearn.decomposition._pca.PCA,
):
    """
    This function should be used for new data to be transformed using an already estimated standardize and
    PCA procedure.

    :param df: the new dataframe of observations to apply transforms to
    :param scaler: sklearn StandardScaler object
    :param pca: sklearn PCA object
    :return: the new standardized and PCA transformed dataframe
    """
    df_transformed = scaler.transform(df)
    df_transformed = pca.transform(df_transformed)
    return df_transformed


def get_num_of_PCs_to_retain(PCA_object: sklearn.decomposition.PCA, PC_variance_explained_min: float):
    cumulative_variances = np.cumsum(PCA_object.explained_variance_ratio_)
    num_of_PCs_to_retain = np.where(cumulative_variances > PC_variance_explained_min)[0][0] + 1
    return num_of_PCs_to_retain, cumulative_variances


# def standardize_and_PCA(df, n_components: Optional[int] = None, PC_variance_explained_min: float = 0.9):
#     scaler = StandardScaler().fit(df)
#     df_transformed = scaler.transform(df)
#     pca = PCA(n_components=n_components).fit(df_transformed)
#     if n_components is None:
#         cumulative_variances = np.cumsum(pca.explained_variance_ratio_)
#         n_components = np.where(cumulative_variances > PC_variance_explained_min)[0][0] + 1
#         print(
#             f"PC's used: {n_components}\nData variation explained: {100 * cumulative_variances[n_components - 1]:.2f} %")
#     df_transformed = pca.transform(df_transformed)
#     df_transformed = df_transformed[:, :n_components]
#     df_transformed = pd.DataFrame(df_transformed,
#                                   columns=[f"PC{i}" for i in range(1, n_components + 1)], index=df.index)
#     return df_transformed, pca, scaler


def plot_features_acf(df, gridsize: Tuple[int, int] = None, nlags: int = 50, corr_type="acf"):
    if corr_type == "acf":
        df_acf = {f"{label}": acf(values, nlags=nlags, fft=True) for label, values in df.items()}
    else:
        df_acf = {f"{label}": pacf(values, nlags=nlags) for label, values in df.items()}
    df_acf = pd.DataFrame(df_acf)
    col_names = df_acf.columns.values
    nrows, ncols = gridsize
    fig, axs = plt.subplots(nrows, ncols, sharex="all", sharey="all")
    col_counter = 0
    # if gridsize is just 1-dimensional, i.e. 1 row or 1 column
    if len(axs.shape) == 1:
        for plot_number in range(max(nrows, ncols)):
            col_name = col_names[col_counter]
            axs[plot_number].stem(df_acf[col_name], linefmt="grey", markerfmt="", bottom=0, basefmt="r--")
            axs[plot_number].set_title(col_name)
            col_counter += 1
    else:
        for row in range(nrows):
            for col in range(ncols):
                col_name = col_names[col_counter]
                axs[row, col].stem(
                    df_acf[col_name],
                    linefmt="grey",
                    markerfmt="",
                    bottom=0,
                    basefmt="r--",
                )
                axs[row, col].set_title(col_name)
                col_counter += 1
    fig.suptitle("Autocorrelation" if corr_type == "acf" else "Partial autocorrelation")
    return fig


class ControlChartPlotMixin:
    def __init__(self):
        self.UCL = None
        self.LCL = None
        self.center_line = None
        self.input_name = None
        self.stat_name = None

    @staticmethod
    def _plot_scalar_or_array(
        x: Union[float, np.ndarray, List[float]],
        ax: matplotlib.axes._subplots.Axes,
        color="red",
    ):
        if isinstance(x, (np.ndarray, pd.Series, list)):
            ax.plot(x, color=color, linestyle="dashed")
        else:
            ax.axhline(x, color=color, linestyle="dashed", label="Control limits")

    def _plot_single_phase_univariate(self, df, y_limit_offsets=(0.95, 1.05)):
        fig, ax = plt.subplots(1, 1)
        df_outside_CL = df.loc[df["outside_CL"], self.stat_name]
        N_samples = df.shape[0]
        ax.plot(
            df[self.stat_name],
            linestyle="-",
            marker="",
            color=single_phase_line_color,
            linewidth=get_line_width(N_samples),
            zorder=1,
        )
        ax.scatter(
            df_outside_CL.index.values,
            df_outside_CL,
            marker="o",
            color="red",
            s=get_outside_CL_marker_size(N_samples),
            zorder=2,
        )
        legend_labels = [self.stat_name]
        if self.center_line is not None:
            ax.axhline(self.center_line, color="blue", alpha=0.7)
            legend_labels.append("Center line")
        if self.UCL is not None:
            self._plot_scalar_or_array(self.UCL, ax, color=UCL_color)
            legend_labels.append("Control limit")
        if self.LCL is not None:
            self._plot_scalar_or_array(self.LCL, ax, color=LCL_color)

        plt.legend(legend_labels, ncol=len(legend_labels))
        y_limits = ax.get_ylim()
        ax.set_ylim(y_limits[0] * y_limit_offsets[0], y_limits[1] * y_limit_offsets[1])
        return fig

    def _plot_single_phase_multivariate(
        self,
        df,
        y_limit_offsets=(0.95, 1.05),
        gridsize: Tuple[int] = None,
        subplot_titles: List[str] = None,
        y_labels: List[str] = None,
    ):

        N_samples = df.shape[0]
        number_of_plots = len(self.stat_name)
        gridsize = (number_of_plots, 1) if gridsize is None else gridsize
        fig, axs = plt.subplots(*gridsize, sharex="all")
        for i in range(number_of_plots):
            stat_to_plot = self.stat_name[i]
            LCL_to_plot = df[f"LCL_{stat_to_plot}"] if f"LCL_{stat_to_plot}" in df.columns else None
            UCL_to_plot = df[f"UCL_{stat_to_plot}"]
            df_outside_CL = df.loc[df[f"outside_CL_{stat_to_plot}"], stat_to_plot]
            axs[i].plot(
                df[stat_to_plot],
                linestyle="-",
                marker="",
                color=single_phase_line_color,
                linewidth=get_line_width(N_samples),
                zorder=1,
            )
            axs[i].scatter(
                df_outside_CL.index.values,
                df_outside_CL,
                s=get_outside_CL_marker_size(N_samples),
                marker="o",
                color="red",
                zorder=2,
            )
            legend_labels = [stat_to_plot]
            if self.center_line is not None:
                axs[i].axhline(self.center_line, color="blue", alpha=0.7)
                legend_labels.append("Center line")
            if UCL_to_plot is not None:
                self._plot_scalar_or_array(UCL_to_plot, axs[i], color=UCL_color)
                legend_labels.append("Control limit")
            if LCL_to_plot is not None:
                self._plot_scalar_or_array(LCL_to_plot, axs[i], color=LCL_color)
            axs[i].legend(legend_labels, ncol=len(legend_labels))
            y_limits = axs[i].get_ylim()
            axs[i].set_ylim(y_limits[0] * y_limit_offsets[0], y_limits[1] * y_limit_offsets[1])
            title = stat_to_plot if subplot_titles is None else subplot_titles[i]
            y_label = "" if y_labels is None else y_labels[i]
            axs[i].set_ylabel(y_label)
            axs[i].set_title(title)
        return fig, axs

    def _plot_two_phases(
        self,
        df_phase1_results: pd.DataFrame,
        df_phase2_results: pd.DataFrame,
        y_limit_offsets=(0.95, 1.05),
    ):
        fig, ax = plt.subplots(1, 1)
        df = pd.concat([df_phase1_results, df_phase2_results])
        df["phase"] = 1
        df["phase"].iloc[len(df_phase1_results) :] = 2
        df = df.reset_index()
        N_samples = df.shape[0]
        df_outside_CL = df.loc[df["outside_CL"], self.stat_name]

        LCL_to_plot = df[f"LCL"] if f"LCL" in df.columns else None
        UCL_to_plot = df[f"UCL"]

        legend_labels = ["Phase 1", "Phase 2"]
        plt.plot(
            df[self.stat_name][df["phase"] == 1],
            linestyle="-",
            marker="",
            color=single_phase_line_color,
            linewidth=get_line_width(N_samples),
            zorder=1,
        )
        plt.plot(
            df[self.stat_name][df["phase"] == 2],
            color=single_phase_line_color,
            linewidth=get_line_width(N_samples),
            linestyle="-",
            marker="",
            zorder=1,
        )
        if self.center_line is not None:
            ax.axhline(self.center_line, color="blue", alpha=0.7)
            legend_labels.append("Center line")
        if UCL_to_plot is not None:
            self._plot_scalar_or_array(UCL_to_plot, ax, color=UCL_color)
            legend_labels.append("Control limit")
        if LCL_to_plot is not None:
            self._plot_scalar_or_array(LCL_to_plot, ax, color=LCL_color)

        ax.scatter(
            df_outside_CL.index.values,
            df_outside_CL,
            s=get_outside_CL_marker_size(N_samples),
            marker="o",
            color="red",
            zorder=2,
        )
        plt.legend(legend_labels, ncol=len(legend_labels))
        y_limits = ax.get_ylim()
        ax.set_ylim(y_limits[0] * y_limit_offsets[0], y_limits[1] * y_limit_offsets[1])
        return fig

    def _plot_two_phases_multivariate(
        self,
        df_phase1_results: pd.DataFrame,
        df_phase2_results: pd.DataFrame,
        y_limit_offsets=(0.95, 1.05),
        gridsize: Tuple[int] = None,
        subplot_titles: List[str] = None,
        y_labels: List[str] = None,
    ):

        index_is_datetime_format = isinstance(self.df_phase1_stats.index, pd.core.indexes.datetimes.DatetimeIndex)
        df = pd.concat([df_phase1_results, df_phase2_results])
        df["phase"] = 1
        df["phase"].iloc[len(df_phase1_results) :] = 2
        if not index_is_datetime_format:
            df = df.reset_index()
        N_samples = df.shape[0]

        number_of_plots = len(self.stat_name)
        gridsize = (number_of_plots, 1) if gridsize is None else gridsize
        fig, axs = plt.subplots(*gridsize, sharex="all")
        for i in range(number_of_plots):
            stat_to_plot = self.stat_name[i]
            LCL_to_plot = df[f"LCL_{stat_to_plot}"] if f"LCL_{stat_to_plot}" in df.columns else None
            UCL_to_plot = df[f"UCL_{stat_to_plot}"]
            df_outside_CL = df.loc[df[f"outside_CL_{stat_to_plot}"], stat_to_plot]
            axs[i].plot(
                df[stat_to_plot][df["phase"] == 1],
                linestyle="-",
                marker="",
                color=multivariate_phase1_color,
                linewidth=get_line_width(N_samples),
                zorder=1,
                label="Phase 1",
            )
            axs[i].plot(
                df[stat_to_plot][df["phase"] == 2],
                linestyle="-",
                marker="",
                color=multivariate_phase2_color,
                linewidth=get_line_width(N_samples),
                zorder=1,
                label="Phase 2",
            )
            axs[i].scatter(
                df_outside_CL.index.values,
                df_outside_CL,
                s=get_outside_CL_marker_size(N_samples),
                marker="o",
                color=outlier_marker_color,
                zorder=2,
            )
            if self.center_line is not None:
                axs[i].axhline(self.center_line, color="blue", alpha=0.7)
            if UCL_to_plot is not None:
                self._plot_scalar_or_array(UCL_to_plot, axs[i], color=UCL_color)
            if LCL_to_plot is not None:
                self._plot_scalar_or_array(LCL_to_plot, axs[i], color=LCL_color)
            axs[i].legend(ncol=2)
            y_limits = axs[i].get_ylim()
            axs[i].set_ylim(y_limits[0] * y_limit_offsets[0], y_limits[1] * y_limit_offsets[1])
            title = stat_to_plot if subplot_titles is None else subplot_titles[i]
            y_label = "" if y_labels is None else y_labels[i]
            axs[i].set_ylabel(y_label)
            axs[i].set_title(title)
            if index_is_datetime_format:
                fig.autofmt_xdate(bottom=0.1, rotation=30, ha="center")
        return fig, axs


def plot_df_acf(df, gridsize: Tuple[int, int] = None, nlags: int = 50, corr_type="acf"):
    if corr_type == "acf":
        df_acf = {f"{label}": acf(values, nlags=nlags, fft=True) for label, values in df.items()}
    else:
        df_acf = {f"{label}": pacf(values, nlags=nlags) for label, values in df.items()}
    df_acf = pd.DataFrame(df_acf)
    col_names = df_acf.columns.values
    nrows, ncols = gridsize
    fig, axs = plt.subplots(nrows, ncols, sharex="all", sharey="all")
    col_counter = 0
    for row in range(nrows):
        for col in range(ncols):
            col_name = col_names[col_counter]
            axs[row, col].stem(df_acf[col_name], linefmt="grey", markerfmt="", bottom=0, basefmt="r--")
            axs[row, col].set_title(col_name)
            col_counter += 1
    fig.suptitle("Autocorrelation" if corr_type == "acf" else "Partial autocorrelation")
    return fig
