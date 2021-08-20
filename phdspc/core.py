from abc import abstractmethod

from phdspc.constants import *
from phdspc.helpers import *


class BaseControlChart(ControlChartPlotMixin):

    def __init__(self, n_sample_size: int):
        self.is_fitted = False
        self.UCL = None
        self.LCL = None
        self.center_line = None
        self.stat_name = None
        self.n_sample_size = n_sample_size

    @abstractmethod
    def fit(self, df_phase1: pd.DataFrame, *args, **kwargs):
        pass

    def _collect_results_df(self, df_stats: pd.DataFrame):
        df_stats_copy = df_stats.copy()
        if self.n_sample_size == 1:
            df_stats_copy = df_stats_copy[[self.input_name, self.stat_name]]
        else:
            df_stats_copy = df_stats_copy[[self.stat_name]]
        df_stats_copy["LCL"] = self.LCL
        df_stats_copy["UCL"] = self.UCL
        df_stats_copy["outside_CL"] = ~df_stats_copy[self.stat_name].between(self.LCL, self.UCL,
                                                                             inclusive="neither")
        return df_stats_copy


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
        super().__init__(n_sample_size=n_sample_size)
        assert self.n_sample_size > 1, "Sample/subgroup size must be greater than 1"
        assert variability_estimator.lower() in ["auto", "range", "std"], "Variance estimator must be one of " \
                                                                          "\"auto\", \"range\" or \"std\""""
        self.stat_name = "sample_mean"
        self.variability_Rbar_or_sbar = None
        self.input_name = None
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
        df_phase1_copy = self._remove_samples_with_only_one_observation(df_phase1_copy)  # std not defined on singletons
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
    Assumes normally distributed, non-autocorrelated data. The R-chart uses an estimate
    of variability based on the in-sample range contrary to the s-chart, which uses the
    standard deviation to estimate variability.
    The R-chart is best suited for samples of size n < 10 or 12.
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


class SChart(RChart):
    """
    Univariate control chart for monitoring the in-sample variation of a quality characteristic.
    Assumes normally distributed, non-autocorrelated data. The s-chart uses an estimate
    of variability based on the in-sample standard deviations contrary to the R-chart, which uses the
    range to estimate variability. The s-chart is better suited for samples of size n > 10 or 12.
    """

    def __init__(self, n_sample_size: int = 5):
        super().__init__(n_sample_size=n_sample_size,
                         variability_estimator="std")


#  TODO implement for sample_size > 1
class MEWMAChart(BaseControlChart, ControlChartPlotMixin):
    """
    - Note, that this is a phase 2 procedure. However, the process target/mean, mu, can reasonably
      be obtained from in-control phase 1 data.
    - Typically used with individual observations, i.e. sample size n = 1, but works for n > 1, too.
    - Small values of lambda seems a good default choice for this procedure, as it makes the procedure:
       1) robust to the underlying distribution, e.g. it is robust to data exhibiting non-normality.
       2) effective in detecting small shifts for small values of lambda.
    - It is recommended to use the EWMA together with a Shewhart chart, e.g. x-bar or Hotelling T^2,
      as the EWMA is slower at detecting large shifts than the Shewhart type chart.
    Based on Montgomery, 2013, chapter 11.4, p. 524.
    """

    def __init__(self, n_sample_size: int = 1, lambda_: float = 0.1, sigma: Optional[np.ndarray] = None):
        super().__init__(n_sample_size=n_sample_size)
        assert self.n_sample_size > 0, "Sample/subgroup size must be greater than 0."
        assert 0.0 < lambda_ <= 1.0, "Bad lambda value given. Lambda must be in the interval 0 < lambda <= 1."
        self.stat_name = "T2"
        self.df_phase2_stats = None
        self.input_dim = None
        self.lambda_ = lambda_
        self.sigma = sigma
        self.df_PCs = None

    def fit(self, df_phase2: pd.DataFrame, *args, **kwargs):
        self.input_dim = df_phase2.shape[1]
        assert self.input_dim > 1, "Multivariate method: number of features must be more than 1. Use the EWMA procedure " \
                                   "for univariate data."
        df_phase2_copy = df_phase2.copy()

        if self.sigma is not None:
            assert self.sigma.shape == (self.input_dim, self.input_dim), "The matrix \"sigma\" must have dimensions " \
                                                                         "equal to the number of input features. " \
                                                                         "E.g. 2 x 2 for 2 input features."
        else:
            self.sigma = df_phase2_copy.corr()  # the correlation matrix

        Z = [np.zeros(shape=self.input_dim)]
        T2 = []
        df_phase2_copy = np.array(df_phase2_copy)
        for i in range(1, df_phase2_copy.shape[0] + 1):
            x_i = df_phase2_copy[i - 1,]
            z_i = self.lambda_ * x_i + (1 - self.lambda_) * Z[i - 1]  # eq: 11.30
            Z.append(z_i)
            sigma_i = (self.lambda_ / (2 - self.lambda_)) * (
                    1 - (1 - self.lambda_) ** (2 * i)) * self.sigma  # eq: 11.32
            T2.append(multiply_matrices(z_i.transpose(), np.linalg.inv(sigma_i), z_i))  # eq: 11.31

        Z.pop(0)
        df_Z = pd.DataFrame(Z, columns=[f"Z{i}" for i in range(1, self.input_dim + 1)])
        self.df_phase2_stats = df_phase2.copy()
        self.df_phase2_stats = pd.concat([self.df_phase2_stats, df_Z], axis=1)
        self.df_phase2_stats[self.stat_name] = T2
        self.is_fitted = True
        return self

    def fit_on_PCs(self, df_phase2: pd.DataFrame, n_components: int = None, PC_variance_explained_min: float = 0.9,
                   *args, **kwargs):
        assert n_components > 1, "Multivariate method: number of features must be more than 1. " \
                                 "Use the EWMA procedure for univariate data."
        df_transformed, pca, _ = standardize_and_PCA(df_phase2, n_components=n_components)
        if n_components is None:
            cumulative_variances = np.cumsum(pca.explained_variance_ratio_)
            n_components = np.where(cumulative_variances < PC_variance_explained_min)[0][0] + 2
        df_transformed = df_transformed[:, :n_components]
        self.sigma = None  # the standardization and PCA changes the original correlations, so we can't use those
        self.df_PCs = pd.DataFrame(df_transformed, columns=[f"PC{i}" for i in range(1, n_components + 1)])
        self.fit(self.df_PCs)

    def compute_delta(self, mu_shifted: np.ndarray):
        """
        This method computes the quantity, delta, of equation 11.33 of Montgomery called the non-centrality parameter.
        It is a measure of a given shift size given by a p x 1 vector, mu, of standard deviations. E.g. for a 6-dimensional
        dataset / process, a vector mu = [1, 1, 1, 1, 1, 1] corresponds to a shift of exactly one standard deviation in each
        feature dimension. Consequently, the zero-vector represents a normal condition.

        :param mu_shifted: the p x 1 shift vector of standard deviations each feature has shifted from the process target.
        :return: the non-centrality parameter, delta = (mu^T * sigma^(-1) * mu)^(1/2). This value measures the multivariate
        size of the shift, all dimensions considered.
        """
        assert self.is_fitted, "No correlation matrix available. Run the fit() method first on suitable phase 2 data."
        mu_shifted_copy = mu_shifted.copy()
        if isinstance(mu_shifted, list):
            mu_shifted_copy = np.array(mu_shifted)
        return np.sqrt(multiply_matrices(mu_shifted_copy.transpose(), np.linalg.inv(self.sigma), mu_shifted_copy))

    def plot_phase2(self):
        """
        Plots the obtained phase 2 statistics for the multivariate EWMA procedure. Requires fit() to have been run first.
        """
        assert self.is_fitted, "No stats to plot. Run fit() first on suitable phase 2 data"
        self._plot_single_phase(self.df_phase2_stats)
        plt.title(fr"Phase 2 MEWMA-chart, $\lambda = {self.lambda_}$")
        plt.ylabel(r"Sample Hotelling $T^2_i$")
        plt.xlabel("Sample [i]")


class HotellingTSquaredChart(MEWMAChart):
    """
    The Hotelling T^2 control chart is a special case of the MEWMA procedure. Namely, for
    lambda = 1, the MEWMA is exactly equivalent to the Hotelling procedure. This value of
    lambda ensures that only the latest sample is used in the calculation of the T^2 statistic.
    """

    def __init__(self, n_sample_size: int = 1, sigma: Optional[np.ndarray] = None):
        super().__init__(n_sample_size=n_sample_size, lambda_=1, sigma=sigma)


# TODO:
#  - adapt to sample size > 1
#  - estimating mu and sigma from given data - good practice?
class EWMAChart(BaseControlChart, ControlChartPlotMixin):
    """The exponentially weighted moving average procedure. Useful for taking autocorrelation into account and better
    than Shewhart type charts at detecting smaller shifts. The EWMA chart is typically used for individual observations,
     i.e. sample sizes = 1.
    A useful combination is an EWMA chart
    of the observations and a Shewhart chart of the residuals from an appropriate time-series model.

    """

    def __init__(self,
                 n_sample_size: int = 1,
                 L_control_limit_width: float = 2.7,
                 lambda_: float = 0.1,
                 mu_process_target: Optional[float] = None,
                 sigma: Optional[float] = None):
        super().__init__(n_sample_size=n_sample_size)
        assert self.n_sample_size > 0, "Sample/subgroup size must be greater than 0."
        assert 0.0 < lambda_ <= 1.0, "Bad lambda value given. Lambda must be in the interval 0 < lambda <= 1."
        self.input_name = None
        self.stat_name = "Z"
        self.df_phase2_stats = None
        self.L_control_limit_width = L_control_limit_width
        self.mu_process_target = mu_process_target
        self.lambda_ = lambda_
        self.sigma = sigma

    def fit(self, df_phase2: pd.DataFrame, *args, **kwargs):
        df_phase2_copy = df_phase2.copy()
        df_phase2_copy = get_df_with_sample_id(df_phase2_copy, n_sample_size=self.n_sample_size)
        self.input_name = df_phase2_copy.columns.values[1]

        if self.lambda_ is None:
            self.lambda_ = np.mean(
                df_phase2_copy[self.input_name])  # estimated process standard deviation, if not given

        if self.sigma is not None:
            assert self.sigma > 0, "Process standard deviation, \"sigma\", must be a positive, real, number > 0."
        else:
            self.sigma = np.std(df_phase2_copy[self.input_name],
                                ddof=1)  # estimated process standard deviation, if not given

        Z = [self.mu_process_target]
        sigma_individual = []
        for i in range(1, df_phase2_copy.shape[0] + 1):
            x_i = df_phase2_copy[self.input_name].iloc[i - 1]
            z_i = self.lambda_ * x_i + (1 - self.lambda_) * Z[i - 1]  # eq: 9.22
            Z.append(z_i)
            sigma_i = np.sqrt(
                self.lambda_ / (2 - self.lambda_) * (1 - (1 - self.lambda_) ** (2 * i)))  # eq: 9.24 & 9.25
            sigma_individual.append(sigma_i)

        sigma_individual = np.array(sigma_individual)
        self.LCL = self.mu_process_target - self.L_control_limit_width * self.sigma * sigma_individual
        self.UCL = self.mu_process_target + self.L_control_limit_width * self.sigma * sigma_individual
        self.center_line = self.mu_process_target

        Z.pop(0)
        self.df_phase2_stats = df_phase2.copy()
        self.df_phase2_stats[self.stat_name] = Z
        self.df_phase2_stats = self._collect_results_df(self.df_phase2_stats)

        self.is_fitted = True
        return self

    def plot_phase2(self):
        """
        Plots the obtained phase 2 statistics for the EWMA procedure. Requires fit() to have been run first.
        """
        assert self.is_fitted, "No stats to plot. Run fit() first on suitable phase 2 data"
        self._plot_single_phase(self.df_phase2_stats)
        plt.title(fr"Phase 2 EWMA-chart, $\lambda = {self.lambda_}$, L = {self.L_control_limit_width}")
        plt.ylabel(r"Sample EWMA value [$Z_i$]")
        plt.xlabel("Sample [i]")


def cusum():
    pass


def pareto_chart():
    pass


def changepoint_model():
    """
    See chapter 10.8, p. 490 og Montgomery
    :return:
    """
    pass


def plot_ACF():
    pass
