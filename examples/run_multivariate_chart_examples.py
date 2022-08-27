import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from spc.core import PCAModelChart

plt.style.use("seaborn")

# fmt: off
df_phase1 = pd.DataFrame(
    {
        "x1": [1, 2, 3, 1, 2, 3, 5, 4, 3, 5, 6, 7, 5, 3, 1, 2, 3, 2, 1, 3, 2],
        "x2": [3, 4, 3, 7, 2, 8, 5, 7, 3, 5, 6, 6, 5, 6, 1, 3, 2, 1, 4, 7, 8],
        "x3": [13, 24, 33, 37, 22, 18, 25, 37, 23, 35, 36, 16, 25, 26, 11, 33, 22, 11, 24, 27, 28],
    }
)
# fmt: on

# np.random.seed(1234)
#
N = 10
df_phase2 = pd.DataFrame(
    dict(
        x1=np.random.normal(size=N, scale=0.1),
        x2=np.random.normal(size=N),
        x3=np.random.normal(size=N),
    )
)


chart = PCAModelChart(n_sample_size=1).fit(
    df_phase1=df_phase1, n_components_to_retain=3, verbose=True
)

chart.predict(df_phase2)

chart.get_contributions(df_phase2)
print(chart.df_contributions)
chart.plot_phase1_and_2(df_phase2)
print(chart.predict(df_phase2))
plt.show()
