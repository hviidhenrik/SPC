import os
from typing import Optional

import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from abc import abstractmethod

from phdspc.helpers import *
from phdspc.constants import *


class BaseControlChart(ControlChartPlotMixin):

    def __init__(self):
        pass

    @abstractmethod
    def fit(self, df_phase1: pd.DataFrame, *args, **kwargs):
        pass


class XBarChart(BaseControlChart):
    """
    Univariate control chart for monitoring the process mean of a quality characteristic.
    Assumes normally distributed, non-autocorrelated data.
    """
    def __init__(self, m_sample_size: int = 2):
        """
        Instantiates the XBarChart object

        :param m_sample_size: number of individual observations in each sample aka rational subgroup size
        """
        super().__init__()
        assert m_sample_size > 1, "Sample/subgroup size must be greater than 1"
        self.UCL = None
        self.LCL = None
        self.center_line = None
        self.Rbar = None
        self.input_name = None
        self.stat_name = "sample_mean"
        self.m_sample_size = m_sample_size
        self.df_phase1_stats = None
        self.df_phase2_stats = None

    def fit(self, df_phase1: pd.DataFrame, *args, **kwargs):
        """
        Estimates the X-bar control limits based on the provided phase 1 data.

        :param df_phase1: a dataframe of "in-control" / healthy / normal data. This is often referred to as phase 1
        :return: the fitted XBarChart object
        """
        df_phase1_copy = df_phase1.copy()
        df_phase1_copy = get_df_with_sample_id(df_phase1_copy, m_sample_size=self.m_sample_size)
        self.input_name = df_phase1_copy.columns.values[1]
        self.df_phase1_stats = get_df_with_sample_means_and_ranges(df_phase1_copy,
                                                                   input_col=self.input_name,
                                                                   grouping_column="sample_id")
        self.center_line = float(self.df_phase1_stats[self.stat_name].mean())  # x double bar
        self.Rbar = float(self.df_phase1_stats["sample_range"].mean())  # R bar

        A2 = get_A2(m_sample_size=self.m_sample_size)

        self.LCL = self.center_line - A2 * self.Rbar
        self.UCL = self.center_line + A2 * self.Rbar
        return self

    def plot_phase1(self):
        """
        Plots phase 1 statistics, in this case the sample averages and the estimated control limits.
        """
        self._plot_single_phase(self.df_phase1_stats)
        plt.title(r"Phase 1 $\bar{X}$-chart")
        plt.ylabel(r"Sample average [$\bar{x}$]")
        plt.xlabel("Sample")

    def plot_phase2(self, df_phase2: pd.DataFrame):
        """
        Plots phase 2 statistics, in this case the sample averages and the estimated control limits.

        :param df_phase2: a dataframe with phase 2 data. Column names must match the phase 1 data given to fit()
        """
        df_phase2_copy = df_phase2.copy()
        if self.df_phase2_stats is None:
            df_phase2_copy = get_df_with_sample_id(df_phase2_copy, m_sample_size=self.m_sample_size)
            self.df_phase2_stats = get_df_with_sample_means_and_ranges(df_phase2_copy,
                                                                       input_col=self.input_name,
                                                                       grouping_column="sample_id")
        self._plot_single_phase(self.df_phase2_stats)
        plt.title(r"Phase 2 $\bar{X}$-chart")
        plt.ylabel(r"Sample average [$\bar{x}$]")
        plt.xlabel("Sample")

    def plot_phase1_and_2(self, df_phase2: pd.DataFrame):
        """
        Plots phase 1 and 2 statistics, in this case the sample averages and the estimated control limits. Phase 1
        and 2 will be displayed in differing colours to easily tell them apart visually.

        :param df_phase2: a dataframe with phase 2 data. Column names must match the phase 1 data given to fit()
        """
        df_phase2_copy = df_phase2.copy()
        df_phase2_copy = get_df_with_sample_id(df_phase2_copy, m_sample_size=self.m_sample_size)

        if self.df_phase2_stats is None:
            self.df_phase2_stats = get_df_with_sample_means_and_ranges(df_phase2_copy,
                                                                       input_col=self.input_name,
                                                                       grouping_column="sample_id")
        self._plot_two_phases(self.df_phase1_stats,
                              self.df_phase2_stats)
        plt.title(r"Phase 1 and 2 $\bar{X}$-chart")
        plt.ylabel(r"Sample average [$\bar{x}$]")

    def plot_control_chart(self,
                           df_phase1: pd.DataFrame = None,
                           df_phase2: pd.DataFrame = None):
        """
        Convenience wrapper that can plot phase 1 or 2 alone as well as both together. The plotted combination is
        inferred from the given dataframes.

        :param df_phase1: a dataframe of "in-control" / healthy / normal data. This is often referred to as phase 1
        :param df_phase2: a dataframe with phase 2 data. Column names must match the phase 1 data given to fit()
        """
        if df_phase1 is not None and df_phase2 is None:
            self.plot_phase1()
        if df_phase1 is None and df_phase2 is not None:
            self.plot_phase2(df_phase2=df_phase2)
        if df_phase1 is not None and df_phase2 is not None:
            self.plot_phase1_and_2(df_phase2=df_phase2)


class Rchart(XBarChart):
    """
    Univariate control chart for monitoring the process variation of a quality characteristic.
    Assumes normally distributed, non-autocorrelated data.
    """
    def fit(self, df_phase1: pd.DataFrame, *args, **kwargs):
        """
        Estimates the R-chart control limits based on the provided phase 1 data.

        :param df_phase1: a dataframe of "in-control" / healthy / normal data. This is often referred to as phase 1.
        :return: the fitted RChart object
        """
        super().fit(df_phase1=df_phase1)
        self.stat_name = "sample_range"
        D4 = get_D4(self.m_sample_size)
        D3 = get_D3(self.m_sample_size)

        self.LCL = D3 * self.Rbar
        self.UCL = D4 * self.Rbar
        self.center_line = self.Rbar
        return self

    def plot_phase1(self):
        """
        Plots phase 1 statistics, in this case the sample averages and the estimated control limits.
        """
        super().plot_phase1()
        plt.title("Phase 1 R-chart")
        plt.ylabel("Sample range [R]")

    def plot_phase2(self, df_phase2: pd.DataFrame):
        """
        Plots phase 2 statistics, in this case the sample averages and the estimated control limits.

        :param df_phase2: a dataframe with phase 2 data. Column names must match the phase 1 data given to fit()
        """
        super().plot_phase2(df_phase2=df_phase2)
        plt.title("Phase 2 R-chart")
        plt.ylabel("Sample range [R]")

    def plot_phase1_and_2(self, df_phase2: pd.DataFrame):
        """
        Plots phase 1 and 2 statistics, in this case the sample averages and the estimated control limits. Phase 1
        and 2 will be displayed in differing colours to easily tell them apart visually.

        :param df_phase2: a dataframe with phase 2 data. Column names must match the phase 1 data given to fit()
        """
        super().plot_phase1_and_2(df_phase2=df_phase2)
        plt.title(r"Phase 1 and 2 R-chart")
        plt.ylabel(r"Sample range [R]")


def pareto_chart():
    pass


def shewhart():
    pass


def hotelling():
    pass


def hotelling_on_PCs():
    pass


def ewma():
    pass


def cusum():
    pass


def changepoint_model():
    """
    See chapter 10.8, p. 490 og Montgomery
    :return:
    """
    pass
