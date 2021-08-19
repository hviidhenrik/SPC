from phdspc.core import *
from phdspc.helpers import *

plt.style.use("seaborn")

df_phase2 = pd.DataFrame({"x1": [1, 2, 3, 1, 2, 3, 5, 4, 3, 5, 6, 7, 5, 3, 1, 2, 3, 2, 1, 3, 2],
                          "x2": [3, 4, 3, 7, 2, 8, 5, 7, 3, 5, 6, 6, 5, 6, 1, 3, 2, 1, 4, 7, 8],
                          "x3": [13, 24, 33, 37, 22, 18, 25, 37, 23, 35, 36, 16, 25, 26, 11, 33, 22, 11, 24, 27, 28],
                          })

# df_phase2 = pd.DataFrame({"x1": [-1.19, 0.12, -1.69, 0.3, 0.89, 0.82, -0.3, 0.63, 1.56, 1.46],
#                           "x2": [0.59, 0.9, 0.4, 0.46, -0.75, 0.98, 2.28, 1.75, 1.58, 3.05]
#                           })

chart = MEWMAChart(lambda_=1, sigma=None).fit(df_phase2=df_phase2)
# H = chart.recommend_control_limit()
chart.compute_delta(np.array([3, 2, 2]))
print(chart.df_phase2_stats)
chart.plot_phase2()
plt.show()

chart.fit_on_PCs(df_phase2=df_phase2, n_components=2)
print(chart.df_phase2_stats)
chart.plot_phase2()
plt.show()

