import os
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error

file = os.path.join("files", "daily-births.csv")

series = read_csv(file, header=0, index_col=0, parse_dates=[0], squeeze=True)

# Create lagged dataset
values = DataFrame(series.values)
dataframe = concat([values.shift(1), values], axis=1)
dataframe.columns = ["t-1", "t+1"]
print(dataframe.head())
