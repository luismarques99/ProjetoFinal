import os
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from matplotlib import pyplot
from pandas.plotting import lag_plot

PATH = os.path.join(".", "week-1_introduction", "1-basic-plots")
os.chdir(PATH)

series = read_csv(os.path.join(
    "files", "daily-minimum-temperatures.csv"), header=0, index_col=0)

# print(series.head())

# series.plot()
# pyplot.show()

# lag_plot(series)
# pyplot.show()

values = DataFrame(series.values)
data_frame = concat([values.shift(1), values], axis=1)
data_frame.columns = ["t-1", "t+1"]
result = data_frame.corr()
print(result)
