import os
from pandas import read_csv
from pandas import DataFrame
from pandas import Grouper
from pandas import concat
from pandas import datetime
from matplotlib import pyplot
from pandas.plotting import lag_plot
from pandas.plotting import autocorrelation_plot

"""
To test any plot, uncomment only the lines of code below the name of it
"""


# In case of running this file in terminal these lines must be uncommented
# This assumes the current working directory is this folder
PATH = os.path.join(".", "week-1_introduction", "1-basic-plots")
os.chdir(PATH)


def parser(x):
    return datetime.strptime(f"190{x}", "%Y-%m")


series = read_csv(os.path.join("files", "shampoo-sales.csv"),
                  header=0,
                  index_col=0,
                  parse_dates=[0],
                  squeeze=True,
                  date_parser=parser)

# Line plots

# Line plot
series.plot()
pyplot.show()

# Black dots plot
# series.plot(style="k.")
# pyplot.show()

# Dashed line plot
# series.plot(style="k-")
# pyplot.show()


# Histogram and density plots

# Histogram plot
# series.hist()
# pyplot.show()

# Density plot
# series.plot(kind="kde")
# pyplot.show()


# Box and whisker plots

# Per year
# groups = series.groupby(Grouper(freq='A'))
# years = DataFrame()
# for name, group in groups:
#     years[name.year] = group.values
# years.boxplot()
# pyplot.show()

# Per month
# one_year = series["1901"]
# groups = one_year.groupby(Grouper(freq="M"))
# months = concat([DataFrame(x[1].values) for x in groups], axis=1)
# months = DataFrame(months)
# months.columns = range(1, 13)
# months.boxplot()
# pyplot.show()


# Heat maps

# Per year
# groups = series.groupby(Grouper(freq="A"))
# years = DataFrame()
# for name, group in groups:
#     years[name.year] = group.values
# years = years.T
# pyplot.matshow(years, interpolation=None, aspect="auto")
# pyplot.show()

# Per month
# one_year = series["1901"]
# groups = one_year.groupby(Grouper(freq="M"))
# months = concat([DataFrame(x[1].values) for x in groups], axis=1)
# months = DataFrame(months)
# months.columns = range(1, 13)
# pyplot.matshow(months, interpolation=None, aspect="auto")
# pyplot.show()


# Lag and scatter plots

# Lag plot
# lag_plot(series)
# pyplot.show()

# Scatter plot
# values = DataFrame(series.values)
# lags = 7
# columns = [values]
# for i in range(1, (lags + 1)):
#     columns.append(values.shift(i))
# data_frame = concat(columns, axis=1)
# columns = ["t+1"]
# for i in range(1, (lags + 1)):
#     columns.append(f"t-{i}")
# data_frame.columns = columns
# pyplot.figure(1)
# for i in range(1, (lags+1)):
#     ax = pyplot.subplot(240+1)
#     ax.set_title(f"t+1 vs t-1 {i}")
#     pyplot.scatter(x=data_frame["t+1"].values, y=data_frame[f"t-{i}"].values)
#     pyplot.show()


# Autocorrelation plot
# autocorrelation_plot(series)
# pyplot.show()
