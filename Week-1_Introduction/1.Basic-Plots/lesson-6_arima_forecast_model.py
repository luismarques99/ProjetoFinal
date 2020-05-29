import os
from pandas import read_csv
from pandas import datetime
from pandas import DataFrame
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA
from matplotlib import pyplot

file = os.path.join("files", "shampoo-sales.csv")


def parser(x):
    return datetime.strptime(f"190{x}", "%Y-%m")


series = read_csv(file, header=0, index_col=0, parse_dates=[0], squeeze=True, date_parser=parser)
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
model = ARIMA(series, order=(5, 1, 0))
model_fit = model.fit(disp=0)
print(model_fit.summary())

# plot residual errors
residuals = DataFrame(model_fit.resid)
residuals.plot()
pyplot.show()
residuals.plot(kind="kde")
pyplot.show()
print(residuals.describe())
