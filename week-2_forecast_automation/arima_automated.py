import os
from pandas import read_csv
from matplotlib import pyplot

"""Void function to set the curent working directory"""


def set_cwd():
    PATH = "week-2_forecast_automation"
    os.chdir(PATH)


"""Void function to set the series"""


def set_series():

    file_path = os.path.join("files", "daily-births.csv")

    series = read_csv(file_path, header=0, index_col=0,
                      parse_dates=False, squeeze=True)


# # p 1-5 / d 0-3 / q 0-3
# for p in range(1, 6):
#     for d in range(4):
#         for q in range(4):

def init():
    set_cwd()
    set_series()
