from phdspc.core import *
from phdspc.helpers import *

plt.style.use("seaborn")

# df_phase1 = pd.DataFrame({"x1": [1, 2, 3, 1, 2, 3, 5, 4, 3, 5, 6, 7, 5, 3, 1, 2, 3, 2, 1, 3, 2]})
# df_phase2 = pd.DataFrame({"x1": [2, 3, 3, 5, 10, 14, 14, ]})
#
# xchart = XBarChart(n_sample_size=3, variability_estimator="std").fit(df_phase1=df_phase1)
# print(xchart.df_phase1_results)
# xchart.plot_phase1()
# plt.show()
#
# xchart = XBarChart(n_sample_size=3, variability_estimator="range").fit(df_phase1=df_phase1)
# print(xchart.df_phase1_results)
# xchart.plot_phase1()
# plt.show()
#
# xchart.plot_phase2(df_phase2=df_phase2)
# plt.show()
#
# xchart.plot_phase1_and_2(df_phase2=df_phase2)
# plt.show()
#
# foo = xchart.get_phase1_results()
# bar = xchart.get_phase2_results(df_phase2)
#
# rchart = RChart(n_sample_size=4, variability_estimator="auto")
# rchart.fit(df_phase1=df_phase1)
# rchart.plot_phase1()
# plt.show()
#
# rchart.plot_phase2(df_phase2=df_phase2)
# plt.show()
#
# rchart.plot_phase1_and_2(df_phase2=df_phase2)
# plt.show()
#
# foo = rchart.get_phase1_results()
# bar = rchart.get_phase2_results(df_phase2)
#
# schart = SChart(n_sample_size=4).fit(df_phase1=df_phase1)
# schart.plot_phase1()
# plt.show()
#
# schart.plot_phase2(df_phase2)
# plt.show()
#
# schart.plot_phase1_and_2(df_phase2)
# plt.show()

df_phase2 = pd.DataFrame(
    {"x1": [9.45, 7.99, 9.29, 11.66, 12.16, 10.18, 8.04, 11.46, 9.2, 10.34, 9.03, 11.47, 10.51, 9.4, 10.08,
            9.37, 10.62, 10.31, 8.52, 10.84, 10.9, 9.33, 12.29, 11.5, 10.6, 11.08, 10.38, 11.62, 11.31, 10.52]})

chart = EWMAChart(lambda_=0.1, mu_process_target=10, sigma=1).fit(df_phase2=df_phase2)
chart.plot_phase2()
plt.show()

