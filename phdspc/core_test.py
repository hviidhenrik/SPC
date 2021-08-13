import numpy as np
import pandas as pd

from phdspc.core import *
from phdspc.helpers import *

plt.style.use("seaborn")

df_phase1 = pd.DataFrame({"x1": [1, 2, 3, 1, 2, 3, 5]})
df_phase2 = pd.DataFrame({"x1": [2, 3, 3, 5, 10, 14, 14]})

# xchart = XBarChart(m_sample_size=2).fit(df_phase1=df_phase1)
# xchart.plot_phase1()
# plt.show()
#
# xchart.plot_phase2(df_phase2=df_phase2)
# plt.show()
#
# xchart.plot_phase1_and_2(df_phase2=df_phase2)
# plt.show()

rchart = Rchart(m_sample_size=2).fit(df_phase1=df_phase1)
rchart.plot_phase1()
plt.show()

rchart.plot_phase2(df_phase2=df_phase2)
plt.show()

rchart.plot_phase1_and_2(df_phase2=df_phase2)
plt.show()
