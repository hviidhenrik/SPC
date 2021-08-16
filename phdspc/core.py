from abc import abstractmethod

from phdspc.constants import *
from phdspc.helpers import *


class BaseControlChart(ControlChartPlotMixin):

    def __init__(self):
        self.is_fitted = False

    @abstractmethod
    def fit(self, df_phase1: pd.DataFrame, *args, **kwargs):
        pass


class XBarChart(BaseControlChart):
    """
    Univariate control chart for monitoring the process mean of a quality characteristic.
    Assumes normally distributed, non-autocorrelated data. This chart goes hand-in-hand with the
    RChart, which estimates process variability.

    Some tips:
    - For phase 1 limits: first establish statistical control in the RChart, as the XBarChart estimates
      are based on the estimate of process variablity. If many points plot outside control limits, it can
      pay off to investigate the patterns of these rather than each individual point.
      Montgomery 2013, chapter 6.2, p. 239.
    - Sample sizes, n, of at least 4 or 5 are usually enough to ensure robustness to normality assumption.
      Do note, that the RChart is more sensitive to non-normal data than the XBarChart
      Montgomery 2013, chapter 6.2.5, p. 254.
    - Control charts based on an estimate of the standard deviation, sigma, is more suitable for
      sample sizes, n > 10 or 12, or n is variable.
      Montgomery 2013, chapter 6.3, p. 259.
    """

    def __init__(self, n_sample_size: int = 5, variability_estimator: str = "auto"):
        """
        Instantiates the XBarChart object

        :param n_sample_size: number of individual observations in each sample aka rational subgroup size
        :param variability_estimator: the estimator to use for data variablity, either "auto", "range"
        or "std" (standard deviation). Std is preferred if n_sample_size > 10. Defaults to "auto", which
        automatically decides which estimator is most appropriate for the situation.
        """
        super().__init__()
        assert n_sample_size > 1, "Sample/subgroup size must be greater than 1"
        assert variability_estimator.lower() in ["auto", "range", "std"], "Variance estimator must be one of " \
                                                                          "\"auto\", \"range\" or \"std\""""
        self.UCL = None
        self.LCL = None
        self.center_line = None
        self.variability_Rbar_or_sbar = None
        self.input_name = None
        self.stat_name = "sample_mean"
        self.n_sample_size = n_sample_size
        self.df_phase1_stats = None
        self.df_phase1_results = None
        self.df_phase2_stats = None
        self.variability_estimator = self._determine_variability_estimator(variability_estimator)

    def fit(self, df_phase1: pd.DataFrame, *args, **kwargs):
        """
        Estimates the X-bar control limits based on the provided phase 1 data.

        :param df_phase1: a dataframe of "in-control" / healthy / normal data. This is often referred to as phase 1
        :return: the fitted XBarChart object
        """
        df_phase1_copy = df_phase1.copy()
        df_phase1_copy = get_df_with_sample_id(df_phase1_copy, n_sample_size=self.n_sample_size)
        self.input_name = df_phase1_copy.columns.values[1]
        df_phase1_copy = self._remove_samples_with_only_one_observation(df_phase1_copy)
        self.df_phase1_stats = self._get_df_with_sample_means_and_variability(df_phase1_copy,
                                                                              grouping_column="sample_id")
        self.center_line = float(self.df_phase1_stats[self.stat_name].mean())  # x double bar
        self.variability_Rbar_or_sbar = float(self.df_phase1_stats["sample_variability"].mean())  # R bar or s bar

        constant_A_number = 2 if self.variability_estimator == "range" else 3
        variability_constant = get_A_constant(A_number=constant_A_number, n_sample_size=self.n_sample_size)

        self.LCL = self.center_line - variability_constant * self.variability_Rbar_or_sbar
        self.UCL = self.center_line + variability_constant * self.variability_Rbar_or_sbar
        self.is_fitted = True
        self.df_phase1_results = self.get_phase1_results()
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
            df_phase2_copy = get_df_with_sample_id(df_phase2_copy, n_sample_size=self.n_sample_size)
            self.df_phase2_stats = self._get_df_with_sample_means_and_variability(df_phase2_copy,
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
        df_phase2_copy = get_df_with_sample_id(df_phase2_copy, n_sample_size=self.n_sample_size)

        if self.df_phase2_stats is None:
            self.df_phase2_stats = self._get_df_with_sample_means_and_variability(df_phase2_copy,
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

    def get_phase1_results(self):
        assert self.is_fitted, "Method parameters not fitted yet. Run .fit() on phase 1 data first."
        self.df_phase1_results = self._collect_results_df(self.df_phase1_stats)
        return self.df_phase1_results

    def get_phase2_results(self, df_phase2: pd.DataFrame):
        df_phase2_results = self._group_samples_and_compute_stats(df_phase2)
        return self._collect_results_df(df_phase2_results)

    def _get_df_with_sample_means_and_variability(self,
                                                  df: pd.DataFrame,
                                                  grouping_column: str = "sample_id"):
        def variability_fun(grp):
            return max(grp) - min(grp) if self.variability_estimator == "range" else np.std(grp, ddof=1)

        df_means_and_variabilities = df.groupby(grouping_column).agg(
            sample_mean=pd.NamedAgg(column=self.input_name, aggfunc=np.mean),
            sample_variability=pd.NamedAgg(column=self.input_name, aggfunc=variability_fun)
        )
        return df_means_and_variabilities

    def _group_samples_and_compute_stats(self, df: pd.DataFrame):
        df_copy = df.copy()
        df_copy = get_df_with_sample_id(df_copy, n_sample_size=self.n_sample_size)
        df_copy = self._get_df_with_sample_means_and_variability(df_copy,
                                                                 grouping_column="sample_id")
        return df_copy

    def _collect_results_df(self, df_stats: pd.DataFrame):
        df_stats_copy = df_stats.copy()
        df_stats_copy = df_stats_copy[[self.stat_name]]
        df_stats_copy["LCL"] = self.LCL
        df_stats_copy["UCL"] = self.UCL
        df_stats_copy["outside_CL"] = ~df_stats_copy[self.stat_name].between(self.LCL, self.UCL,
                                                                             inclusive="neither")
        return df_stats_copy

    def _determine_variability_estimator(self, variability_estimator: str):
        if variability_estimator.lower() == "auto":
            return "std" if self.n_sample_size > 10 else "range"
        else:
            return variability_estimator.lower()

    def _remove_samples_with_only_one_observation(self, df_with_sample_id: pd.DataFrame):
        """
        This function removes samples with only one observation to guard against 0 or nans
        in the estimation of variability by either range or standard deviation
        :param df_with_sample_id: a dataframe grouped by sample_id
        :return: the same dataframe, but with size 1 samples removed
        """
        df_grouped = df_with_sample_id.groupby("sample_id").count()
        samples_to_keep = df_grouped[df_grouped[self.input_name] > 1].index.values
        return df_with_sample_id[df_with_sample_id["sample_id"].isin(samples_to_keep)]


class RChart(XBarChart):
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
        self.stat_name = "sample_variability"

        constant_B3_or_D3 = get_B_constant(B_number=3, n_sample_size=self.n_sample_size)
        constant_B4_or_D4 = get_B_constant(B_number=4, n_sample_size=self.n_sample_size)
        if self.variability_estimator == "range":
            constant_B3_or_D3 = get_D_constant(D_number=3, n_sample_size=self.n_sample_size)
            constant_B4_or_D4 = get_D_constant(D_number=4, n_sample_size=self.n_sample_size)

        self.LCL = constant_B3_or_D3 * self.variability_Rbar_or_sbar
        self.UCL = constant_B4_or_D4 * self.variability_Rbar_or_sbar
        self.center_line = self.variability_Rbar_or_sbar
        self.is_fitted = True
        return self

    def plot_phase1(self):
        """
        Plots phase 1 statistics, in this case the sample ranges and the estimated control limits.
        """
        super().plot_phase1()
        chart_type = "R" if self.variability_estimator == "range" else "s"
        unit = "range" if self.variability_estimator == "range" else "Std. dev."
        plt.title(f"Phase 1 {chart_type}-chart")
        plt.ylabel(f"Within sample variation [{unit}]")

    def plot_phase2(self, df_phase2: pd.DataFrame):
        """
        Plots phase 2 statistics, in this case the sample ranges and the estimated control limits.

        :param df_phase2: a dataframe with phase 2 data. Column names must match the phase 1 data given to fit()
        """
        super().plot_phase2(df_phase2=df_phase2)
        chart_type = "R" if self.variability_estimator == "range" else "s"
        unit = "range" if self.variability_estimator == "range" else "Std. dev."
        plt.title(f"Phase 2 {chart_type}-chart")
        plt.ylabel(f"Within sample variation [{unit}]")

    def plot_phase1_and_2(self, df_phase2: pd.DataFrame):
        """
        Plots phase 1 and 2 statistics, in this case the sample averages and the estimated control limits. Phase 1
        and 2 will be displayed in differing colours to easily tell them apart visually.

        :param df_phase2: a dataframe with phase 2 data. Column names must match the phase 1 data given to fit()
        """
        super().plot_phase1_and_2(df_phase2=df_phase2)
        chart_type = "R" if self.variability_estimator == "range" else "s"
        unit = "range" if self.variability_estimator == "range" else "Std. dev."
        plt.title(f"Phase 1 and 2 {chart_type}-chart")
        plt.ylabel(f"Within sample variation [{unit}]")


class EWMA():
    pass


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
