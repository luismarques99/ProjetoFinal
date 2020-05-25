import os
from pandas import read_csv
from matplotlib import pyplot

series = read_csv(os.path.join('files', 'daily-births.csv'), header=0, index_col=0)

# pandas .head() default lines are 5, but it can be passed as an argument [.head(10)]
print(f'\nHead:\n{series.head()}')

print(f'\nSize: {series.size}')

# Query
query = series.query('Date == "1959-01-15"')
print(f'\nQuery:\n{query}')

# Summary statistics
print(f'\nSummary statistics:\n{series.describe()}')

# Show graph
pyplot.plot(series)
pyplot.show()
