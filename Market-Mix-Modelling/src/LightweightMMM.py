# pip install jax==0.3.13 https://whls.blob.core.windows.net/unstable/cuda111/jaxlib-0.3.7+cuda11.cudnn82-cp38-none-win_amd64.whl
# pip install --upgrade git+https://github.com/google/lightweight_mmm.git


import pandas as pd
from lightweight_mmm import preprocessing, lightweight_mmm, plot, optimize_media
import jax.numpy as jnp
from sklearn.metrics import mean_absolute_error
from tabulate import tabulate

# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
# pd.set_option('display.width', 1000)
# pd.set_option('expand_frame_repr', True)
# pd.options.display.max_seq_items = None
# pd.set_option('display.max_colwidth', None)
# pd.set_option('display.colheader_justify', 'center')

pd.set_option('display.max_colwidth', -1)

data = pd.read_excel("Market-Mix-Modelling\data\Advertising Sample.xlsx")
data.head()
data.columns






