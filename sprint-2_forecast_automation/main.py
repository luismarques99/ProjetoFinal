import os
from arima_automated import arima_model
from pandas import datetime
from modules.csv_writer import csv_writer

# PATH = os.path.join(".", "sprint-2_forecast_automation")
# os.chdir(PATH)


def parser(x):
    return datetime.strptime(f"190{x}", "%Y-%m")


# for p in range(1, 6):
#     for d in range(4):
#         for q in range(4):
#             arima_model(filename="shampoo-sales.csv", date_parser=parser, p=p, d=d, q=q)

# FIXME: Resolver problema dos modules. Não consigo importar
writer = csv_writer("teste.csv", ["param1", "param2", "param3"])
writer.write_line(["boas", "123", '"list of strings included"'])
writer.write_line(["boas", "123", '"list of strings included"'])
writer.write_line(["boas", "123", '"list of strings included"'])
writer.write_line(["boas", "123", '"list of strings included"'])
writer.write_line(["boas", "123", '"list of strings included"'])
writer.close()
