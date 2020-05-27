import os
import re
from pandas import read_csv
from pandas import DataFrame
from pandas import Grouper
from pandas import concat
from matplotlib import pyplot

PATH = os.path.join(".", "Week-1_Introduction", "1.Basic-Plots")
os.chdir(PATH)

series = read_csv(os.path.join("files", "shampoo-sales.csv"),
                  header=0, index_col=0, parse_dates=True, squeeze=True)
series_b = read_csv(os.path.join("files", "daily-minimum-temperatures.csv"), header=0, index_col=0, parse_dates=True,
                    squeeze=True)


# Line plots

# Line plot
# series_b.plot()
# pyplot.show()

# Black dots plot
# series.plot(style="k.")
# pyplot.show()

# Dashed line plot
# series.plot(style="k-")
# pyplot.show()


# Histogram and density plots

# Histogram plot
# series_b.hist()
# pyplot.show()

# Density plot
# series.plot(kind="kde")
# pyplot.show()


# Box and whisker plots

# FIXME:
# Tive de fazer com outro dataset, porque o fornecido não traz a data como é esperado (o ano vem definido como ('1', '2', '3'))
# Vou ter de fazer um groupby especifico para que o dataset shampoo-sales.csv funcione neste tipo de gráficos

# Per year
# groups = series_b.groupby(Grouper(freq='A'))
# years = DataFrame()
# for name, group in groups:
#     years[name.year] = group.values
# years.boxplot()
# pyplot.show()

# Per month
# one_year = series_b["1990"]
# groups = one_year.groupby(Grouper(freq="M"))
# months = concat([DataFrame(x[1].values) for x in groups], axis=1)
# months = DataFrame(months)
# months.columns = range(1, 13)
# months.boxplot()
# pyplot.show()


# Heat maps
groups = series_b.groupby(Grouper(freq="A"))
years = DataFrame()
for name, group in groups:
    years[name.year] = group.values
years = years.T
pyplot.matshow(years, interpolation=None, aspect="auto")
pyplot.show()
