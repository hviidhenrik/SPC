from spc.core import *
from spc.helpers import *

plt.style.use("seaborn")
path = "C:/datasets/SSV feedwater pumps/all tags/SPC/"
df_phase1 = pd.read_csv(path + "data_pump_30_phase1_2015-2017_A.csv", index_col="timelocal")
df_phase2 = pd.read_csv(path + "data_pump_30_phase2_2017_A.csv", index_col="timelocal")

df_phase1 = df_phase1.sample(frac=0.2).sort_index()
df_phase2 = df_phase2.sample(frac=0.2).sort_index()

df_phase1.index = pd.to_datetime(df_phase1.index)
df_phase2.index = pd.to_datetime(df_phase2.index)

# df_phase2["temp_slipring_diff"].plot()
# plt.show()

chart = PCAModelChart(alpha=0.005).fit(
    df_phase1, n_components_to_retain=None, PC_variance_explained_min=0.9, verbose=True
)
# print(chart.df_T2_contributions)
# print(chart.df_Q_contributions)
# chart = HotellingT2Chart().fit(df_phase1)
# chart.plot_phase1()
# plt.show()
#
# chart.plot_phase2(df_phase2)
# plt.show()

fig, axs = chart.plot_phase1_and_2(df_phase2)
plt.show()
# plt.savefig("test.svg")
