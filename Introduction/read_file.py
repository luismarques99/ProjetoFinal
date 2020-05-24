from pandas import read_csv

series = read_csv('daily-births.csv', header=0, index_col=0)
series.head()
