# pip install jax==0.3.13 https://whls.blob.core.windows.net/unstable/cuda111/jaxlib-0.3.7+cuda11.cudnn82-cp38-none-win_amd64.whl
# pip install --upgrade git+https://github.com/google/lightweight_mmm.git


"""
The modeling approach is called Bayesian Time-Varying Coefficients (BTVC) and its available on Orbit, their forecasting package, as Kernel Time-Varying Regression.
We will use Impressions as the features and Sales as the target.
Impressions are how many times someone saw at least 50% of an ad. It can be the same person seeing it multiple times.

MaxAbsScaler simply divides each column value by the maximal absolute value it can find on the column.

1. Read the dataset
2. Divide into three datasets (both train and test) - Impressions, Costs and Sales(target) and scale them
3. Finding The Right Hyper-Parameters
4. Train the model with right hyper-parameters
5. Analyze the media effects and the ROI as well as budget optimizer

"""


import pandas as pd
from lightweight_mmm import preprocessing, lightweight_mmm, plot, optimize_media
import jax.numpy as jnp
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import mean_absolute_error

pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)

data = pd.read_excel("Market-Mix-Modelling\data\Advertising Sample.xlsx")
data.head()
data.columns

data.dtypes
data['Date'].min()                                                                                  # Timestamp('2021-10-17 00:00:00')
data['Date'].max()                                                                                  # Timestamp('2021-01-11 00:00:00')
data['Date'].max() - data['Date'].min()                                                             # 86 days

data['Ad group alias'].nunique()                                                                    # 20
data['Ad group alias'].unique()
data['Brands'] = data['Ad group alias'].str[:7]
data['Ad_groups'] = data['Ad group alias'].str[8:]
data['Brands'].value_counts()
data['Ad_groups'].value_counts().sort_values(ascending=True)

agg_data = data.groupby(['Date', 'Ad group alias'])[['Impressions', 'Spend' , 'Sales']].sum()
agg_data.head()
agg_data.shape                                                                                      # (1415, 3)

agg_data = agg_data.drop(["Brand 1 Ad Group 12"], axis=0, level=1)
agg_data.head()
agg_data.shape                                                                                      # (1411, 3)

media_data_raw = agg_data['Impressions'].unstack().fillna(0); media_data_raw.head()
costs_raw = agg_data['Spend'].unstack();costs_raw.head()
sales_raw = agg_data['Sales'].reset_index().groupby("Date").sum(); sales_raw.head()


split_point = -28                                                                                   # 28 days to end of data

media_data_train = media_data_raw.iloc[:split_point].copy()
media_data_test = media_data_raw.iloc[split_point:].copy()
media_data_raw.shape                                                                                # (87, 19)
media_data_train.shape                                                                              # (59, 19)
media_data_test.shape                                                                               # (28, 19)
media_data_train.head()
media_data_test.head()

target_train = sales_raw.iloc[:split_point].copy()
target_test = sales_raw.iloc[split_point:].copy()
target_train.shape                                                                                  # (59, 1)
target_test.shape                                                                                   # (28, 1)
target_train.head()
target_test.head()

costs_train = costs_raw.iloc[:split_point].copy().sum(axis=0).values.reshape(-1,1)
costs_train.shape                                                                                   # (19,1)
costs_train[:1]
# costs_train.columns

# media_scaler = preprocessing.CustomScaler(divide_operation=jnp.mean)
# split_point = pd.Timestamp("2021-12-15") # 28 days to end of data
# media_data_train = media_data_raw.loc[:split_point - pd.Timedelta(1,'D')]


scaler_x = MaxAbsScaler()
scaler_y = MaxAbsScaler()
media_data_train_scaled = scaler_x.fit_transform(media_data_train.values)
media_data_test_scaled = scaler_x.transform(media_data_test.values)
costs_train_scaled = scaler_x.fit_transform(costs_train)
target_train_scaled = scaler_y.fit_transform(target_train.values)

media_data_train_scaled[:1]
media_data_test_scaled[:1]
target_train_scaled[:1]
costs_train_scaled[:1]

media_names = media_data_raw.columns

# Finding The Right Hyper-Parameters
adstock_models = ["adstock", "hill_adstock", "carryover"]
degrees_season = [1,2,3]

mmm = lightweight_mmm.LightweightMMM(model_name="adstock")
mmm.fit(
    media = media_data_train_scaled,
    total_costs = costs_train_scaled,
    target = target_train_scaled,
    number_warmup = 1000,
    number_samples = 1000,
    number_chains = 1,
    degrees_seasonality = 1,
    weekday_seasonality = True,
    seasonality_frequency = 365,
    seed = 1,
    # XLA_FLAGS=--xla_gpu_force_compilation_parallelism=1
)