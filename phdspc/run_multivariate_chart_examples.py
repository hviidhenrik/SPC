import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from phdspc.core import *
from phdspc.helpers import *

plt.style.use("seaborn")

df_phase1 = pd.DataFrame({"x1": [1, 2, 3, 1, 2, 3, 5, 4, 3, 5, 6, 7, 5, 3, 1, 2, 3, 2, 1, 3, 2],
                          "x2": [3, 4, 3, 7, 2, 8, 5, 7, 3, 5, 6, 6, 5, 6, 1, 3, 2, 1, 4, 7, 8],
                          "x3": [13, 24, 33, 37, 22, 18, 25, 37, 23, 35, 36, 16, 25, 26, 11, 33, 22, 11, 24, 27, 28],
                          })
# np.random.seed(1234)
N = 50
df_phase2 = pd.DataFrame(dict(x1=np.random.normal(size=N, scale=0.1),
                              x2=np.random.normal(size=N),
                              x3=np.random.normal(size=N)))


# chart = HotellingT2Chart(n_sample_size=1).fit(df_phase1=df_phase1, verbose=True)
# print(chart.df_phase1_stats)
# chart.plot_phase1()
# plt.show()


chart = PCAModelChart(n_sample_size=1).fit(df_phase1=df_phase1, n_components_to_retain=2, verbose=True)
# chart = HotellingT2Chart(n_sample_size=1).fit(df_phase1=df_phase1, verbose=True)
chart.plot_phase1()
chart.plot_phase2(df_phase2)
chart.plot_phase1_and_2(df_phase2)
plt.show()













# path = "C:/Users/HEHHA/OneDrive - Ørsted/Desktop/datasets/SSV feedwater pumps/divided into phases/"
# df_phase1 = pd.read_csv(path + "data_pump30_phase1.csv", index_col="timelocal")
# df_phase2 = pd.read_csv(path + "data_pump30_phase2.csv", index_col="timelocal")
# df_phase2_meta = df_phase2[["sample", "faulty"]]
# df_phase1 = df_phase1.drop(columns=["sample", "faulty", "effect_pump_20_MW"])
# df_phase2 = df_phase2.drop(columns=["sample", "faulty", "effect_pump_20_MW"])
# df_phase2 = df_phase2.sample(frac=0.8)

# plot_acf(df_phase1["temp_slipring_diff"], lags=1000)
# plt.show()
#
# plot_pacf(df_phase1["temp_slipring_diff"], lags=100)
# plt.show()

# df_phase2 = pd.DataFrame({"x1": [-1.19, 0.12, -1.69, 0.3, 0.89, 0.82, -0.3, 0.63, 1.56, 1.46],
#                           "x2": [0.59, 0.9, 0.4, 0.46, -0.75, 0.98, 2.28, 1.75, 1.58, 3.05]
#                           })

# # H = chart.recommend_control_limit()
# print(chart.df_phase2_stats)
# chart.plot_phase2()
# plt.show()
#
# chart = MEWMAChart(lambda_=0.1, sigma=None)
# chart.fit_on_PCs(df_phase1=df_phase1, df_phase2=df_phase2, n_components=None, PC_variance_explained_min=0.99,
#                  verbose=True)
# print(chart.df_phase2_stats)
# chart.plot_phase2()
# plt.show()

