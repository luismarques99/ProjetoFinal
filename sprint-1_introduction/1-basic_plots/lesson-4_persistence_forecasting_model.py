import os
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error

PATH = os.path.join(".", "week-1_introduction", "1-basic_plots")
os.chdir(PATH)

file = os.path.join("files", "daily-births.csv")

series = read_csv(file, header=0, index_col=0, parse_dates=[0], squeeze=True)

# Create lagged dataset
values = DataFrame(series.values)
data_frame = concat([values.shift(1), values], axis=1)
data_frame.columns = ["t-1", "t+1"]
print(data_frame.head())

# Split into train and test sets
X = data_frame.values
train_size = int(len(X) * 0.66)
train, test = X[1: train_size], X[train_size:]
train_X, train_y = train[:, 0], train[:, 1]
test_X, test_y = test[:, 0], test[:, 1]


# Persistence model
def model_persistence(x):
    return x


# Walk-forward validation
predictions = list()
for x in test_X:
    yhat = model_persistence(x)
    predictions.append(yhat)
test_score = mean_squared_error(test_y, predictions)
print(f"Test MSE: {format(test_score, '0.3f')}")

# Plot predictions and expected results
pyplot.plot(train_y)
pyplot.plot([None for i in train_y] + [x for x in test_y])
pyplot.plot([None for i in train_y] + [x for x in predictions])
pyplot.show()
