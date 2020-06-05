import os
from pandas import read_csv
from matplotlib import pyplot

PATH = os.path.join(".", "Week-1_Introduction", "1.Basic-Plots")
os.chdir(PATH)

series = read_csv(os.path.join(
    "files", "daily-minimum-temperatures.csv"), header=0, index_col=0)

print(series.head())
series.plot()
pyplot.show()
