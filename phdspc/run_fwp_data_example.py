import pandas as pd

from phdspc.core import *
from phdspc.helpers import *

plt.style.use("seaborn")

# path = "C:/Users/HEHHA/OneDrive - Ørsted/Desktop/datasets/SSV feedwater pumps/divided into phases/data_pump30_phase2.csv"
path = "C:/projects/smartplant/sp-pdm-phd-ARGUE/data/ssv_feedwater_pump/data_pump30_phase2.csv"
df_phase2 = pd.read_csv(path, index_col="timelocal")
df_phase2_meta = df_phase2[["sample", "faulty"]]
df_phase2["temp_slipring_diff"].plot()
plt.show()

# run MEWMA on all variables
df_phase2_all = df_phase2.drop(columns=["sample", "faulty", "effect_pump_20_MW"])
# df_phase2_all = df_phase2.sample(frac=)


l_values = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
for l in l_values:
    chart = MEWMAChart(lambda_=l, sigma=None)
    chart.fit_on_PCs(df_phase2=df_phase2_all, n_components=None, PC_variance_explained_min=0.95, verbose=True)
    print(chart.df_phase2_stats)
    chart.plot_phase2()
    plt.show()


# run EWMA on only bearing_temp_diff variable

# df_phase2_diff = df_phase2[["temp_slipring_diff"]]
# df_phase2_diff = df_phase2_diff.sample(frac=0.1)
#
# chart = EWMAChart(lambda_=0.1, sigma=None)
# chart.fit(df_phase2=df_phase2_diff)
# print(chart.df_phase2_stats)
# chart.plot_phase2()
# plt.show()