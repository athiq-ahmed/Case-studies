# pip install jax==0.3.13 https://whls.blob.core.windows.net/unstable/cuda111/jaxlib-0.3.7+cuda11.cudnn82-cp38-none-win_amd64.whl
# pip install --upgrade git+https://github.com/google/lightweight_mmm.git


import pandas as pd
from lightweight_mmm import preprocessing, lightweight_mmm, plot, optimize_media
import jax.numpy as jnp
from sklearn.metrics import mean_absolute_error
from tabulate import tabulate

# pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
# pd.set_option('display.width', 1000)
# pd.set_option('expand_frame_repr', True)
# pd.options.display.max_seq_items = None
# pd.set_option('display.max_colwidth', None)
# pd.set_option('display.colheader_justify', 'center')

pd.set_option('display.max_colwidth', -1)

data = pd.read_excel("Market-Mix-Modelling\data\Advertising Sample.xlsx")
data.head()
data.columns

data.dtypes
data['Date'].min()    # Timestamp('2021-10-17 00:00:00')
data['Date'].max()    # Timestamp('2021-01-11 00:00:00')

data['Ad group alias'].nunique()  # 20
data['Ad group alias'].unique()
data['Brands'] = data['Ad group alias'].str[:7]
data['Ad_groups'] = data['Ad group alias'].str[8:]
data['Brands'].value_counts()
data['Ad_groups'].value_counts().sort_values(ascending=True)

agg_data = data.groupby(['Date', 'Ad group alias'])[['Impressions', 'Spend' , 'Sales']].sum()
agg_data.head()
agg_data.shape      #(1415, 3)

agg_data = agg_data.drop(["Brand 1 Ad Group 12"], axis=0, level=1)
agg_data.head()
agg_data.shape      #(1411, 3)

media_data_raw = agg_data['Impressions'].unstack().fillna(0); media_data_raw.head()
costs_raw = agg_data['Spend'].unstack();costs_raw.head()
sales_raw = agg_data['Sales'].reset_index().groupby("Date").sum(); sales_raw.head()
media_data_raw['Sales'] = sales_raw
media_data_raw = media_data_raw.reset_index()
media_data_raw.head()

# MaxAbsScaler simply divides each column value by the maximal absolute value it can find on the column.


