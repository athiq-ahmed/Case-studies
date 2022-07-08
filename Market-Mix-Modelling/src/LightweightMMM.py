# pip install --upgrade git+https://github.com/google/lightweight_mmm.git

import pandas as pd
from lightweight_mmm import preprocessing, lightweight_mmm, plot, optimize_media
import jax.numpy as jnp
from sklearn.metrics import mean_absolute_error

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', -1)
# pd.set_option('expand_frame_repr', False)
pd.options.display.max_seq_items = None

data = pd.read_excel("data/Advertising Sample.xlsx")
data.head()








# https://forecastegy.com/posts/implementing-uber-marketing-mix-model-with-orbit/
# https://www.thewindowsclub.com/how-to-enable-or-disable-win32-long-paths-in-windows-11-10