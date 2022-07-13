# pip install jax==0.3.13 https://whls.blob.core.windows.net/unstable/cuda111/jaxlib-0.3.7+cuda11.cudnn82-cp38-none-win_amd64.whl
# pip install --upgrade git+https://github.com/google/lightweight_mmm.git
# orbit requires some installations - https://code.visualstudio.com/docs/cpp/config-mingw

"""
The modeling approach is called Bayesian Time-Varying Coefficients (BTVC) and its available on Orbit, their forecasting package, as Kernel Time-Varying Regression.
We will use Impressions as the features and Sales as the target.
"""



import pandas as pd
from lightweight_mmm import preprocessing, lightweight_mmm, plot, optimize_media
import jax.numpy as jnp
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import mean_absolute_error
import orbit

# from orbit.models import KTR
# from orbit.models import DLT
# from orbit.models.dlt import DLTFull
# from orbit.models.dlt import ETSFull, DLTMAP, DLTFull
# from orbit.models.lgt import LGTMAP, LGTFull, LGTAggregated
# from orbit.diagnostics.plot import plot_predicted_components

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
media_data_raw.head(10)

media_data_raw['Date'].min()    # Timestamp('2021-10-17 00:00:00')
media_data_raw['Date'].max()    # Timestamp('2022-01-11 00:00:00')
media_data_raw['Date'].max() - media_data_raw['Date'].min()  # 86 days


# MaxAbsScaler simply divides each column value by the maximal absolute value it can find on the column.
media_data_raw.columns
date_col = 'Date'
response_col = 'Sales'

regressor_cols = ['Brand 1 Ad Group 1', 'Brand 1 Ad Group 10', 'Brand 1 Ad Group 11', 'Brand 1 Ad Group 13', 'Brand 1 Ad Group 2', 'Brand 1 Ad Group 3',
                 'Brand 1 Ad Group 4', 'Brand 1 Ad Group 5', 'Brand 1 Ad Group 6', 'Brand 1 Ad Group 7', 'Brand 1 Ad Group 8', 'Brand 1 Ad Group 9',
                 'Brand 2 Ad Group 1', 'Brand 2 Ad Group 2', 'Brand 2 Ad Group 3', 'Brand 2 Ad Group 4', 'Brand 2 Ad Group 5', 'Brand 1 Ad Group 14', 
                 'Brand 2 Ad Group 6']


split_point = -28 # 28 days to end of data

media_data_train = media_data_raw.iloc[:split_point].copy()
media_data_test = media_data_raw.iloc[split_point:].copy()

media_data_raw.shape    # (87, 21)
media_data_train.shape  # (59, 21)
media_data_test.shape   # (28, 21)

media_data_train.head()
media_data_test.head()

scaler_x = MaxAbsScaler()
media_data_train.loc[:, regressor_cols] = scaler_x.fit_transform(media_data_train.loc[:, regressor_cols])
media_data_test.loc[:, regressor_cols] = scaler_x.transform(media_data_test.loc[:, regressor_cols])

scaler_y = MaxAbsScaler()
media_data_train.loc[:, response_col] = scaler_y.fit_transform(media_data_train.loc[:, response_col].values.reshape(-1,1))

media_names = media_data_raw.columns


# Building the model
ktr_ = KTR(
    response_col = response_col,
    date_col = date_col,
    regressor_col = regressor_cols,
    seed = 2022,
    seasonality = [7],
    estimator = 'pyro-svi', 
    n_bootstrap_draws = 1e4,
    num_steps = 301,
    message = 100
)

ktr_.fit(media_data_train)
predicted_df = ktr_.predict(media_data_test)

# Estimating Media Effects
ktr_.plot_regression_coefs(figsize=(10, 10), include_ci=True)

# Estimating Return on Investment (ROI)
contribution = scaler_y.inverse_transform(ktr_.get_regression_coefs().iloc[:, 1:] * media_data_full[regressor_cols])
contribution = pd.DataFrame(contribution,columns=regressor_cols).sum(axis=0)
roi = (contribution.sum(axis=0) / costs_raw.sum(axis=0)).clip(0)

