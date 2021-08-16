import numpy as np
import pandas as pd
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


class ControlChartPlotMixin:

    def __init__(self):
        self.UCL = None
        self.LCL = None
        self.center_line = None
        self.input_name = None
        self.stat_name = None

    def _plot_single_phase(self, df):
        fig, ax = plt.subplots(1, 1)
        ax.plot(df[self.stat_name], linestyle="-", marker="o", color="black")
        if self.LCL is not None:
            ax.axhline(self.LCL, color="red", linestyle="dashed", label="Control limits")
        if self.UCL is not None:
            ax.axhline(self.UCL, color="red", linestyle="dashed")
        if self.center_line is not None:
            ax.axhline(self.center_line, color="blue", alpha=0.7, label="Center line")

        plt.legend(ncol=2)
        y_limits = ax.get_ylim()
        ax.set_ylim(y_limits[0]-3, y_limits[1]+3)
        return fig

    def _plot_two_phases(self, df_phase1: pd.DataFrame, df_phase2: pd.DataFrame):
        fig, ax = plt.subplots(1, 1)
        df = pd.concat([df_phase1, df_phase2])
        df["phase"] = 1
        df["phase"].iloc[len(df_phase1):] = 2
        df = df.reset_index()

        plt.plot(df[self.stat_name][df["phase"] == 1],
                 linestyle="-", marker="o", color="green", label="Phase 1")
        plt.plot(df[self.stat_name][df["phase"] == 2],
                 linestyle="-", marker="o", color="orange", label="Phase 2")
        plt.axhline(self.center_line, color="blue", alpha=0.7, label="Center line")
        plt.axhline(self.LCL, color="red", linestyle="dashed", label="Control limits")
        plt.axhline(self.UCL, color="red", linestyle="dashed")
        plt.legend(ncol=2)
        y_limits = ax.get_ylim()
        ax.set_ylim(y_limits[0]-3, y_limits[1]+3)
        return fig
