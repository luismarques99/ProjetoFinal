import os
from arima_automated import arima_model
from pandas import datetime

PATH = os.path.join(".", "sprint-2_forecast_automation")
os.chdir(PATH)


def parser(x):
    return datetime.strptime(f"190{x}", "%Y-%m")


for p in range(1, 6):
    for d in range(4):
        for q in range(4):
            arima_model(filename="shampoo-sales.csv", date_parser=parser, p=p, d=d, q=q)
