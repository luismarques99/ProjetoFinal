import os
import re
from pandas import read_csv
from pandas import DataFrame
from pandas import Grouper
from matplotlib import pyplot

CWD = os.path.join(".", "semana-1_Introduction", "1.Basic-Plots")
os.chdir(CWD)

series = read_csv(os.path.join("files", "shampoo-sales.csv"),
                  header=0, index_col=0, parse_dates=True, squeeze=True)
series_b = read_csv(os.path.join("files", "daily-minimum-temperatures.csv"), header=0, index_col=0, parse_dates=True,
                    squeeze=True)


# Line plots
# Line plot
series_b.plot()
# Black dots plot
# series.plot(style="k.")
# Dashed line plot
# series.plot(style="k-")
pyplot.show()


# Histogram and density plots
# Histogram plot
series.hist()
# Density plot
# series.plot(kind="kde")
pyplot.show()


# Box and whisker plots
# FIXME:
# Tive de fazer com outro dataset, porque o fornecido não traz a data como é esperado (o ano vem definido como ('1', '2', '3'))
# Vou ter de fazer um groupby especifico para que o dataset shampoo-sales.csv funcione neste tipo de gráficos

# pattern = re.compile("1-")
# for key in series.keys():
# 	if pattern.match(key):
# 		print(key)

groups = series_b.groupby(Grouper(freq='A'))
years = DataFrame()
for name, group in groups:
    years[name.year] = group.values
years.boxplot()
pyplot.show()


# Heat maps
