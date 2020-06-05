import os
from os.path import dirname, join
from pandas import read_csv
from matplotlib import pyplot

# In case of running this file in terminal these lines must be uncommented
# This assumes the current working directory is this folder
PATH = os.path.join(".", "Week-1_Introduction", "1.Basic-Plots")
os.chdir(PATH)

series = read_csv(join("files", "daily-births.csv"),
                  header=0, index_col=0, parse_dates=True)

# pandas .head() default lines are 5, but it can be passed as an argument [.head(10)]
print(f"\nHead:\n{series.head()}")

print(f"\nSize: {series.size}")

# Query
query = series.query("Date == '1959-01-15'")
print(f"\nQuery:\n{query}")
# print(series['1959'])

# Summary statistics
print(f"\nSummary statistics:\n{series.describe()}")

# Show graph
pyplot.plot(series)
pyplot.show()
