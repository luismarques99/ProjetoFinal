import os
from pandas import read_csv
from pandas import DataFrame
from pandas import Grouper
from matplotlib import pyplot

series = read_csv(os.path.join("files", "shampoo-sales.csv"), header=0, index_col=0, parse_dates=True)

# Line plots
# Line plot
series.plot()
# Black dots plot
series.plot(style="k.")
# Dashed line plot
series.plot(style="k-")

# Histogram and density plots
# Histogram plot
series.hist()
# Density plot
series.plot(kind="kde")

# Box and whisker plots
# groups = series.groupby(Grouper(freq='A'))
# years = DataFrame()
# for name, group in groups:
#     years[name.year] = group.values
# years.boxplot()

pyplot.show()
