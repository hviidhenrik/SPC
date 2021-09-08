from phdspc.core import *
from phdspc.helpers import *

plt.style.use("seaborn")
path = "C:/datasets/SSV feedwater pumps/all tags/SPC/"
df_phase1 = pd.read_csv(path + "data_pump_30_phase1_2015-2017.csv", index_col="timelocal")
df_phase2 = pd.read_csv(path + "data_pump_30_phase2_2017.csv", index_col="timelocal")

df_phase1.index = pd.to_datetime(df_phase1.index)
df_phase2.index = pd.to_datetime(df_phase2.index)

# df_phase2["temp_slipring_diff"].plot()
# plt.show()

chart = PCAModelChart().fit(df_phase1, n_components_to_retain=None, PC_variance_explained_min=0.9, verbose=True)
# chart = HotellingT2Chart().fit(df_phase1)
# chart.plot_phase1()
# plt.show()

chart.plot_phase2(df_phase2)
plt.show()

chart.plot_phase1_and_2(df_phase2)
plt.show()
