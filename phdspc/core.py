from abc import abstractmethod

from scipy.stats import beta, chi2, f

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
        df_stats_copy["outside_CL"] = ~df_stats_copy[self.stat_name].between(self.LCL, self.UCL, inclusive="neither")
        return df_stats_copy


class XBarChart(BaseControlChart):
    """
    Univariate control chart for monitoring the process mean of a quality characteristic.
    Assumes normally distributed, non-autocorrelated data. This chart goes hand-in-hand with the
    RChart, which estimates process variability.

    Some tips:
    - For phase 1 limits: first establish statistical control in the RChart, as the XBarChart estimates
      are based on the estimate of process variablity. If many samples plot outside control limits, it can
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
        assert variability_estimator.lower() in ["auto", "range", "std"], (
            "Variance estimator must be one of " '"auto", "range" or "std"' ""
        )
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
        self.df_phase1_stats = self._get_df_with_sample_means_and_variability(
            df_phase1_copy, grouping_column="sample_id"
        )
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
        self._plot_single_phase_univariate(self.df_phase1_results)
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
            self.df_phase2_stats = self._get_df_with_sample_means_and_variability(
                df_phase2_copy, grouping_column="sample_id"
            )
        df_phase2_results = self._collect_results_df(self.df_phase2_stats)
        self._plot_single_phase_univariate(df_phase2_results)
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
            # TODO: need to refactor this: the object should not have attributes related to phase 2 data
            self.df_phase2_stats = self._get_df_with_sample_means_and_variability(
                df_phase2_copy, grouping_column="sample_id"
            )

        df_phase2_results = self._collect_results_df(self.df_phase2_stats)
        self._plot_two_phases(self.df_phase1_results, df_phase2_results)
        plt.title(r"Phase 1 and 2 $\bar{X}$-chart")
        plt.ylabel(r"Sample average [$\bar{x}$]")

    def plot_control_chart(self, df_phase1: pd.DataFrame = None, df_phase2: pd.DataFrame = None):
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

    def _get_df_with_sample_means_and_variability(self, df: pd.DataFrame, grouping_column: str = "sample_id"):
        def variability_fun(grp):
            return max(grp) - min(grp) if self.variability_estimator == "range" else np.std(grp, ddof=1)

        df_means_and_variabilities = df.groupby(grouping_column).agg(
            sample_mean=pd.NamedAgg(column=self.input_name, aggfunc=np.mean),
            sample_variability=pd.NamedAgg(column=self.input_name, aggfunc=variability_fun),
        )
        return df_means_and_variabilities

    def _group_samples_and_compute_stats(self, df: pd.DataFrame):
        df_copy = df.copy()
        df_copy = get_df_with_sample_id(df_copy, n_sample_size=self.n_sample_size)
        df_copy = self._get_df_with_sample_means_and_variability(df_copy, grouping_column="sample_id")
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
        self.df_phase1_results = self._collect_results_df(self.df_phase1_stats)
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
        super().__init__(n_sample_size=n_sample_size, variability_estimator="std")


#  TODO implement for sample_size > 1
#    - implement MEWMA on residual PC's control chart
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

    def __init__(
            self,
            n_sample_size: int = 1,
            lambda_: float = 0.1,
            sigma: Optional[np.ndarray] = None,
    ):
        super().__init__(n_sample_size=n_sample_size)
        assert self.n_sample_size > 0, "Sample/subgroup size must be greater than 0."
        assert 0.0 < lambda_ <= 1.0, "Bad lambda value given. Lambda must be in the interval 0 < lambda <= 1."
        self.stat_name = "T2"
        self.df_phase2_stats = None
        self.input_dim = None
        self.lambda_ = lambda_
        self.sigma = sigma
        self.df_PCs = None

    def fit(self, df_phase2: pd.DataFrame, verbose: bool = False, *args, **kwargs):
        self.input_dim = df_phase2.shape[1]
        assert self.input_dim > 1, (
            "Multivariate method: number of features must be more than 1. Use the EWMA procedure "
            "for univariate data."
        )
        df_phase2_copy = df_phase2.copy()

        if self.sigma is not None:
            assert self.sigma.shape == (self.input_dim, self.input_dim), (
                'The matrix "sigma" must have dimensions '
                "equal to the number of input features. "
                "E.g. 2 x 2 for 2 input features."
            )
        else:
            self.sigma = df_phase2_copy.corr()  # the correlation matrix

        Z = [np.zeros(shape=self.input_dim)]
        T2 = []
        df_phase2_copy = np.array(df_phase2_copy)
        N_rows = df_phase2_copy.shape[0]
        for i in range(1, N_rows + 1):
            if verbose and i % np.ceil(0.1 * N_rows) == 0:
                print(f"Progress: {100 * i / (df_phase2_copy.shape[0] + 1):.2f} %")
            x_i = df_phase2_copy[
                i - 1,
            ]
            z_i = self.lambda_ * x_i + (1 - self.lambda_) * Z[i - 1]  # eq: 11.30
            Z.append(z_i)
            sigma_i = (
                    (self.lambda_ / (2 - self.lambda_)) * (1 - (1 - self.lambda_) ** (2 * i)) * self.sigma
            )  # eq: 11.32
            T2.append(multiply_matrices(z_i.transpose(), np.linalg.inv(sigma_i), z_i))  # eq: 11.31

        Z.pop(0)
        df_Z = pd.DataFrame(
            Z,
            columns=[f"Z{i}" for i in range(1, self.input_dim + 1)],
            index=df_phase2.index,
        )
        self.df_phase2_stats = df_phase2.copy()
        self.df_phase2_stats = pd.concat([self.df_phase2_stats, df_Z], axis=1)
        self.df_phase2_stats[self.stat_name] = T2
        self.is_fitted = True
        return self

    def fit_on_PCs(
            self,
            df_phase1: pd.DataFrame,
            df_phase2: pd.DataFrame,
            n_components: int = None,
            PC_variance_explained_min: float = 0.9,
            verbose: bool = False,
            *args,
            **kwargs,
    ):
        df_transformed, pca, scaler = standardize_and_PCA(df_phase1, n_components=n_components)
        if n_components is None:
            cumulative_variances = np.cumsum(pca.explained_variance_ratio_)
            n_components = np.where(cumulative_variances > PC_variance_explained_min)[0][0] + 1
        assert n_components > 1, (
            "Multivariate method: number of features must be more than 1. "
            "Use the EWMA procedure for univariate data."
        )
        if verbose:
            print(f"Number of PC's used: {n_components}")
        df_phase2_transformed = apply_standardize_and_PCA(df_phase2, scaler=scaler, pca=pca)
        df_phase2_transformed = df_phase2_transformed[:, :n_components]
        # TODO try to estimate sigma on phase 1 data
        self.sigma = None  # the standardization and PCA changes the original correlations, so we can't use those
        self.df_PCs = pd.DataFrame(
            df_phase2_transformed,
            columns=[f"PC{i}" for i in range(1, n_components + 1)],
        )
        self.fit(self.df_PCs, verbose=verbose)

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
        self._plot_single_phase_univariate(self.df_phase2_stats)
        plt.title(fr"Phase 2 MEWMA-chart, $\lambda = {self.lambda_}$")
        plt.ylabel(r"Sample Hotelling $T^2_i$")
        plt.xlabel("Sample [i]")


# TODO: adapt to n_sample_size > 1 at some point
class HotellingT2Chart(BaseControlChart, ControlChartPlotMixin):
    def __init__(self, n_sample_size: int = 1, alpha: float = 0.05):
        super().__init__(n_sample_size=n_sample_size)
        self.alpha = alpha
        self.stat_name = "T2"
        self.input_dim = None
        self.sigma = None
        self.sigma_inverse = None
        self.x_bar = None
        self.m_samples = None
        self.df_phase1_stats = None
        self.df_contributions = None

    def fit(
            self,
            df_phase1: pd.DataFrame,
            compute_contributions: bool = True,
            verbose=False,
            *args,
            **kwargs,
    ):
        self.m_samples = df_phase1.shape[0]
        self.input_dim = df_phase1.shape[1]
        x = np.array(df_phase1)
        self.x_bar = np.mean(x, axis=0)
        self.sigma = np.cov(x.T, ddof=1)  # n0te: should the S_5 estimator be used instead? Are they equivalent?
        self.sigma_inverse = np.linalg.inv(self.sigma)
        T2 = []
        for i in range(x.shape[0]):
            T2.append(self._compute_T2_value(x[i]))

        self.UCL = self._compute_T2_UCL(phase=1)
        self.df_phase1_stats = pd.DataFrame(
            dict(
                T2=T2,
                UCL=self.UCL,
                outside_CL=T2 > self.UCL,
                cumulated_prop_outside_CL=np.cumsum(1 * (T2 > self.UCL)) / self.m_samples,
            )
        )
        self.is_fitted = True

        if compute_contributions and self.input_dim > 2:
            # compute contributions by removing the k'th variable, to see how much it contributes to the T^2 value of a sample
            contributions = self._compute_T2_contributions(T2, x)
            self.df_contributions = pd.DataFrame(contributions, columns=df_phase1.columns, index=df_phase1.index)

        return self

    def _compute_T2_contributions(self, T2, x):
        contributions = np.zeros_like(x, dtype=float)
        for k in range(self.input_dim):
            x_k = np.delete(x, k, axis=1)
            x_bar_k = np.delete(self.x_bar, k)
            sigma_inverse_k = np.delete(np.delete(self.sigma, k, axis=0), k, axis=1)
            sigma_inverse_k = np.linalg.inv(sigma_inverse_k)
            for i in range(x_k.shape[0]):
                contributions[i, k] = T2[i] - self._compute_T2_value_given_xbar_and_sigma_inv(
                    x_k[i], x_bar_k, sigma_inverse_k
                )
        return contributions

    def predict(self, df_phase2: pd.DataFrame):
        """
        Calculates T^2 statistics for new data and returns a dataframe with the results.
        n0te: may need a better name for this method

        :param df_phase2: dataframe with new data
        :return: dataframe with the calculated T^2 results and control limits
        """
        T2 = [self._compute_T2_value(x) for _, x in df_phase2.iterrows()]
        UCL = self._compute_T2_UCL(phase=2)
        return pd.DataFrame(
            dict(
                T2=T2,
                UCL=UCL,
                outside_CL=T2 > UCL,
                cumulated_prop_outside_CL=np.cumsum(1 * (T2 > UCL)) / self.m_samples,
            )
        )

    def _compute_T2_value(self, x: np.ndarray):
        assert x.shape == self.x_bar.shape, f"The shape of x ({x.shape}) must be the same as x_bar: {self.x_bar.shape}"
        return multiply_matrices((x - self.x_bar).T, self.sigma_inverse, (x - self.x_bar))

    @staticmethod
    def _compute_T2_value_given_xbar_and_sigma_inv(
            x: np.ndarray, x_bar: np.ndarray = None, sigma_inverse: np.ndarray = None
    ):
        assert x.shape == x_bar.shape, f"The shape of x ({x.shape}) must be the same as x_bar: {x_bar.shape}"
        return multiply_matrices((x - x_bar).T, sigma_inverse, (x - x_bar))

    def _compute_T2_UCL(self, phase: Union[int, str]):
        p, m, n = self.input_dim, self.m_samples, self.n_sample_size
        if int(phase) == 1:
            percentile = beta.ppf(q=1 - self.alpha, a=p / 2, b=(m - p - 1) / 2)
            self.UCL = (m - 1) ** 2 / m * percentile
        else:
            percentile = f.ppf(q=1 - self.alpha, dfn=p, dfd=m - p)
            self.UCL = (p * (m + 1) * (m - 1)) / (m * m - m * p) * percentile
        return self.UCL

    def plot_phase1(self):
        """
        Plots phase 1 statistics, in this case the sample averages and the estimated control limits.
        """
        self._plot_single_phase_univariate(self.df_phase1_stats)
        final_proportion_outside_CL = self.df_phase1_stats["cumulated_prop_outside_CL"].tail(1).squeeze()
        plt.suptitle(
            f"Phase 1 Hotelling $T^2$-chart,\n $\\alpha$ = {100 * self.alpha:.2f} %, "
            f"samples outside CL = {100 * final_proportion_outside_CL:.2f} %"
        )
        plt.ylabel(r"Sample $T^2$")
        plt.xlabel("Sample")

    def plot_phase2(self, df_phase2: pd.DataFrame):
        """
        Plots phase 2 statistics, in this case the sample averages and the estimated control limits.

        :param df_phase2: a dataframe with phase 2 data. Column names must match the phase 1 data given to fit()
        """
        df_phase2_stats = self.predict(df_phase2)
        self._plot_single_phase_univariate(df_phase2_stats)
        final_proportion_outside_CL = df_phase2_stats["cumulated_prop_outside_CL"].tail(1).squeeze()
        plt.suptitle(
            f"Phase 2 Hotelling $T^2$-chart,\n $\\alpha$ = {100 * self.alpha:.2f} %, "
            f"samples outside CL = {100 * final_proportion_outside_CL:.2f} %"
        )
        plt.ylabel(r"Sample $T^2$")
        plt.xlabel("Sample")

    def plot_phase1_and_2(self, df_phase2: pd.DataFrame):
        """
        Plots phase 1 and 2 statistics, in this case the sample averages and the estimated control limits. Phase 1
        and 2 will be displayed in differing colours to easily tell them apart visually.

        :param df_phase2: a dataframe with phase 2 data. Column names must match the phase 1 data given to fit()
        """

        df_phase2_results = self.predict(df_phase2)
        self._plot_two_phases(self.df_phase1_stats, df_phase2_results)
        plt.title(r"Phase 1 and 2 $\bar{X}$-chart")
        plt.ylabel(r"Sample average [$\bar{x}$]")


class PCAModelChart(HotellingT2Chart):
    def __init__(self, n_sample_size: int = 1, alpha: float = 0.05):
        super().__init__(n_sample_size=n_sample_size, alpha=alpha)
        self.scaler = None
        self.PCA = None
        self.UCL_Q = None
        self.n_components_to_retain = None
        self.loadings = None
        self.scores = None
        self.df_T2_contributions = None
        self.df_Q_contributions = None

    def fit(
            self,
            df_phase1: pd.DataFrame,
            n_components_to_retain: int = None,
            PC_variance_explained_min: float = 0.9,
            compute_contributions: bool = True,
            verbose=False,
            *args,
            **kwargs,
    ):
        self.n_components_to_retain = n_components_to_retain
        self.m_samples = df_phase1.shape[0]
        df_PCA_transformed, self.PCA, self.scaler = standardize_and_PCA(df_phase1)
        self.stat_name = (self.stat_name, "Q")
        if self.n_components_to_retain is None:
            self.n_components_to_retain, cumulative_variances = get_num_of_PCs_to_retain(
                self.PCA, PC_variance_explained_min
            )
            vprint(
                verbose,
                f"PC's used: {self.n_components_to_retain}\nData variation explained: "
                f"{100 * cumulative_variances[self.n_components_to_retain - 1]:.2f} %",
            )

        # first call parent class to estimate T^2 statistics
        df_PCA_transformed = df_PCA_transformed[:, : self.n_components_to_retain]
        PC_names = [f"PC{i}" for i in range(1, df_PCA_transformed.shape[1] + 1)]
        super().fit(df_phase1=pd.DataFrame(df_PCA_transformed, columns=PC_names, index=df_phase1.index))
        self.is_fitted = False  # overrides is_fitted attribute set by parent's fit
        self.df_phase1_stats = self.df_phase1_stats.rename(
            columns={
                "UCL": "UCL_T2",
                "outside_CL": "outside_CL_T2",
                "cumulated_prop_outside_CL": "cumulated_prop_outside_CL_T2",
            }
        )

        # 2nd, calculate Q statistic and its control limit
        Q = self._compute_Q_values(df_phase1)  # PCA is done inside this function, so Q is done on the PC's
        self.df_phase1_stats["Q"] = Q
        self.UCL_Q = self._compute_Q_UCL(Q)
        df_Q_stats = pd.DataFrame(
            dict(
                UCL_Q=self.UCL_Q,
                outside_CL_Q=Q > self.UCL_Q,
                cumulated_prop_outside_CL_Q=np.cumsum(1 * (Q > self.UCL_Q)) / df_phase1.shape[0],
            )
        )
        self.df_phase1_stats = pd.concat([self.df_phase1_stats, df_Q_stats], axis=1)
        self.df_phase1_stats.index = df_phase1.index

        if compute_contributions:
            # n0te: should _compute_contributions be called on df_PCA_transformed instead?
            Q_contributions, T2_contributions = self._compute_contributions(df_phase1)
            self.df_T2_contributions = pd.DataFrame(T2_contributions, columns=df_phase1.columns, index=df_phase1.index)
            self.df_Q_contributions = pd.DataFrame(Q_contributions, columns=df_phase1.columns, index=df_phase1.index)

        self.is_fitted = True
        return self

    def predict(self, df_phase2: pd.DataFrame):
        df_transformed = apply_standardize_and_PCA(df_phase2, self.scaler, self.PCA)
        df_transformed = pd.DataFrame(df_transformed[:, : self.n_components_to_retain])
        df_phase2_stats = super().predict(df_transformed)
        Q = self._compute_Q_values(df_phase2)
        df_phase2_stats["Q"] = Q
        df_Q_stats = pd.DataFrame(
            dict(
                UCL_Q=self.UCL_Q,
                outside_CL_Q=Q > self.UCL_Q,
                cumulated_prop_outside_CL_Q=np.cumsum(1 * (Q > self.UCL_Q)) / df_phase2.shape[0],
            )
        )
        df_phase2_stats = df_phase2_stats.rename(
            columns={
                "UCL": "UCL_T2",
                "outside_CL": "outside_CL_T2",
                "cumulated_prop_outside_CL": "cumulated_prop_outside_CL_T2",
            }
        )
        df_phase2_stats = pd.concat([df_phase2_stats, df_Q_stats], axis=1)
        df_phase2_stats.index = df_phase2.index
        return df_phase2_stats

    def get_contributions(self, df: pd.DataFrame):
        """
        Computes contributions for an arbitrary dataframe (phase 1 or 2). Requires fit() to be run first,
        as principal components and scalers need to be estimated.

        :param df: the data observations for the given phase
        :return: dataframe of contributions
        """
        assert self.is_fitted, "Model not fitted. Run fit() method first."
        df_transformed = apply_standardize_and_PCA(df, self.scaler, self.PCA)
        df_transformed = pd.DataFrame(df_transformed[:, : self.n_components_to_retain])
        Q_contributions, T2_contributions = self._compute_contributions(df_transformed)
        T2_col_names = ["T2_contribution_" + col for col in df.columns]
        Q_col_names = ["Q_contribution_" + col for col in df.columns]
        df_T2_contributions = pd.DataFrame(T2_contributions, columns=T2_col_names, index=df.index)
        df_Q_contributions = pd.DataFrame(Q_contributions, columns=Q_col_names, index=df.index)
        return pd.concat([df_T2_contributions, df_Q_contributions], axis=1)

    def _compute_contributions(self, df_phase1):
        input_dim = df_phase1.shape[1]
        T2_contributions = np.zeros_like(df_phase1, dtype=float)
        Q_contributions = np.zeros_like(df_phase1, dtype=float)
        A_eigenvals = np.diag(self.PCA.explained_variance_[: self.n_components_to_retain])
        C_loadings = np.array(self.loadings)
        X = self.scaler.transform(df_phase1)
        X_means = np.mean(X, axis=0)
        T2_matrix_scaling = multiply_matrices(C_loadings, np.linalg.inv(np.sqrt(A_eigenvals)), C_loadings.T)  # p x p
        Q_matrix_scaling = np.eye(input_dim) - multiply_matrices(C_loadings, C_loadings.T)  # p x p
        for i in range(X.shape[0]):
            residual = X[i] - X_means  # 1 x p
            T2_contributions[i] = multiply_matrices(residual, T2_matrix_scaling)  # 1 x p
            Q_contributions[i] = multiply_matrices(residual, Q_matrix_scaling)  # 1 x p
        return Q_contributions, T2_contributions

    def _compute_Q_values(self, df_raw: pd.DataFrame):
        # n0te: most of the next bit is from Max's code. Original source?
        self.scores = apply_standardize_and_PCA(df_raw, self.scaler, self.PCA)[:, : self.n_components_to_retain]
        self.loadings = self.PCA.components_.T[:, : self.n_components_to_retain]  # loadings from phase 1 data
        X = self.scaler.transform(df_raw)
        Xhat = multiply_matrices(self.scores, self.loadings.T)
        Q = np.sum((X - Xhat) ** 2, axis=1)  # n0te: squared prediction error or E matrix in the slides. Is this right?
        return Q

    def _compute_Q_UCL(self, Q: np.ndarray):
        g = np.var(Q, ddof=1) / (2 * np.mean(Q))  # scale factor
        h = (2 * np.mean(Q) ** 2) / np.var(Q, ddof=1)  # degrees of freedom
        percentile = chi2.ppf(q=1 - self.alpha, df=h)
        self.UCL_Q = g * percentile
        return self.UCL_Q

    def plot_phase1(self):
        """
        Plots phase 1 statistics, in this case the sample averages and the estimated control limits.
        """
        final_proportions_outside_CL = (
            self.df_phase1_stats[["cumulated_prop_outside_CL_T2", "cumulated_prop_outside_CL_Q"]].tail(1).squeeze()
        )
        self._plot_single_phase_multivariate(
            self.df_phase1_stats,
            subplot_titles=[
                f"$T^2$-chart, samples outside CL: " f"{100 * final_proportions_outside_CL[0]:.2f} %",
                f"Q-chart, samples outside CL: " f"{100 * final_proportions_outside_CL[1]:.2f} %",
            ],
            y_labels=["Sample $T^2$", "Sample Q"],
        ),
        plt.suptitle(f"Phase 1, $\\alpha$ = {100 * self.alpha:.2f} %")
        if not isinstance(self.df_phase1_stats.index, pd.core.indexes.datetimes.DatetimeIndex):
            plt.xlabel("Sample")

    def plot_phase2(self, df_phase2: pd.DataFrame):
        """
        Plots phase 2 statistics, in this case the sample averages and the estimated control limits.

        :param df_phase2: a dataframe with phase 2 data. Column names must match the phase 1 data given to fit()
        """
        df_phase2_stats = self.predict(df_phase2)

        final_proportions_outside_CL = (
            df_phase2_stats[["cumulated_prop_outside_CL_T2", "cumulated_prop_outside_CL_Q"]].tail(1).squeeze()
        )
        self._plot_single_phase_multivariate(
            df_phase2_stats,
            subplot_titles=[
                f"$T^2$-chart, samples outside CL: " f"{100 * final_proportions_outside_CL[0]:.2f} %",
                f"Q-chart, samples outside CL: " f"{100 * final_proportions_outside_CL[1]:.2f} %",
            ],
            y_labels=["Sample $T^2$", "Sample Q"],
            y_limit_offsets=(0.8, 1.2),
        )
        plt.suptitle(f"Phase 2, $\\alpha$ = {100 * self.alpha:.2f} %")
        if not isinstance(self.df_phase1_stats.index, pd.core.indexes.datetimes.DatetimeIndex):
            plt.xlabel("Sample")

    def plot_phase1_and_2(self, df_phase2: pd.DataFrame):
        """
        Plots phase 1 and 2 statistics, in this case the sample averages and the estimated control limits. Phase 1
        and 2 will be displayed in differing colours to easily tell them apart visually.

        :param df_phase2: a dataframe with phase 2 data. Column names must match the phase 1 data given to fit()
        """
        df_phase2_results = self.predict(df_phase2)
        T2_outlier_percentage = 100 * df_phase2_results["cumulated_prop_outside_CL_T2"].tail(1).squeeze()
        Q_outlier_percentage = 100 * df_phase2_results["cumulated_prop_outside_CL_Q"].tail(1).squeeze()
        T_title = f"$T^2$-chart\n$\\alpha$ = {100 * self.alpha:.2f} %, phase 2 samples outside control limit: {T2_outlier_percentage:.2f} %"
        Q_title = f"Q-chart\n$\\alpha$ = {100 * self.alpha:.2f} %, phase 2 samples outside control limit: {Q_outlier_percentage:.2f} %"
        fig, axs = self._plot_two_phases_multivariate(
            self.df_phase1_stats,
            df_phase2_results,
            subplot_titles=[T_title, Q_title],
            y_labels=["Sample $T^2$", "Sample Q"],
        )
        fig.subplots_adjust(top=0.88)
        plt.suptitle(f"Phase 1 and 2 PCA SPC model")
        fig.tight_layout()
        return fig, axs


# TODO:
#  - adapt to sample size > 1
class EWMAChart(BaseControlChart, ControlChartPlotMixin):
    """The exponentially weighted moving average procedure. Useful for taking autocorrelation into account and better
    than Shewhart type charts at detecting smaller shifts. The EWMA chart is typically used for individual observations,
     i.e. sample sizes = 1.
    A useful combination is an EWMA chart
    of the observations and a Shewhart chart of the residuals from an appropriate time-series model.

    """

    def __init__(
            self,
            n_sample_size: int = 1,
            L_control_limit_width: float = 2.7,
            lambda_: float = 0.1,
            mu_process_target: Optional[float] = None,
            sigma: Optional[float] = None,
    ):
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
                df_phase2_copy[self.input_name]
            )  # estimated process standard deviation, if not given

        if self.sigma is not None:
            assert self.sigma > 0, 'Process standard deviation, "sigma", must be a positive, real, number > 0.'
        else:
            self.sigma = np.std(
                df_phase2_copy[self.input_name], ddof=1
            )  # estimated process standard deviation, if not given

        if self.mu_process_target is None:
            self.mu_process_target = np.mean(df_phase2_copy[self.input_name])  # estimated process mean, if not given

        Z = [self.mu_process_target]
        sigma_individual = []
        for i in range(1, df_phase2_copy.shape[0] + 1):
            x_i = df_phase2_copy[self.input_name].iloc[i - 1]
            z_i = self.lambda_ * x_i + (1 - self.lambda_) * Z[i - 1]  # eq: 9.22
            Z.append(z_i)
            sigma_i = np.sqrt(
                self.lambda_ / (2 - self.lambda_) * (1 - (1 - self.lambda_) ** (2 * i))
            )  # eq: 9.24 & 9.25
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
        self._plot_single_phase_univariate(self.df_phase2_stats)
        plt.title(fr"Phase 2 EWMA-chart, $\lambda = {self.lambda_}$, L = {self.L_control_limit_width}")
        plt.ylabel(r"Sample EWMA value [$Z_i$]")
        plt.xlabel("Sample [i]")


def cusum():
    pass


def pareto_chart():
    """
    See chapter 5.4, p. 208 of Montgomery
    :return:
    """
    pass


def changepoint_model():
    """
    See chapter 10.8, p. 490 og Montgomery
    :return:
    """
    pass


def plot_ACF():
    pass


def plot_contributions():
    pass
