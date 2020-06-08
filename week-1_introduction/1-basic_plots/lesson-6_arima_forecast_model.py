import os
from pandas import read_csv
from pandas import datetime
from pandas import DataFrame
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot

PATH = os.path.join(".", "Week-1_Introduction", "1.Basic-Plots")
os.chdir(PATH)

file_path = os.path.join("files", "shampoo-sales.csv")


def parser(x):
    return datetime.strptime(f"190{x}", "%Y-%m")


series = read_csv(file_path, header=0, index_col=0, parse_dates=0,
                  squeeze=True, date_parser=parser)
# print(series.head())

# Running this example tells us the time series is not stationary and require differencing to make
# it stationary at least a difference order of 1 (d = 1)
# series.plot()
# pyplot.show()

# Running this example we can see that there is a positive correlation with the first 10-to-12 lags
# this is perhaps significant for the first 5 lags. A good starting point may be 5 (p = 5)
# autocorrelation_plot(series)
# pyplot.show()

# fit model
# model = ARIMA(series, order=(5, 1, 0))
# model_fit = model.fit(disp=0)
# print(model_fit.summary())

# plot residual errors
# residuals = DataFrame(model_fit.resid)
# residuals.plot()
# pyplot.show()
# residuals.plot(kind="kde")
# pyplot.show()
# print(residuals.describe())

# Rolling forecast ARIMA model
X = series.values
size = int(len(X) * 0.66)
train, test = X[0: size], X[size: len(X)]
history = [x for x in train]
predictions = list()

for t in range(len(test)):
    model = ARIMA(history, order=(5, 1, 0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    print(f"predicted={yhat}, expected={obs}")

# Mean squared error of the predictions
error = mean_squared_error(test, predictions)
print(f"Test MSE: {format(error, '0.3f')}")

# plot
# Blue is the train dataset
# Green is the expected dataset from predictions
# Red is the actual predictions
pyplot.plot(train)
pyplot.plot([None for i in train] + [x for x in test], color="green")
pyplot.plot([None for i in train] + [x for x in predictions], color="red")
pyplot.show()
