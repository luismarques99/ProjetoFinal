import os
import sys

from datetime import datetime
from matplotlib import pyplot
from pandas import read_csv
from pandas import concat
from pandas import Grouper
from pandas import DataFrame

ROOT_PATH = os.path.join(os.path.dirname(os.path.abspath(__name__)))
PATH = os.path.join(ROOT_PATH, "src")

"""Adds the root folder to the sys path"""
sys.path.append(ROOT_PATH)

"""Sets the current folder as the current working directory"""
os.chdir(PATH)

# def parser(x):
#     return datetime.strptime(x, "%d/%m/%Y")

DATASETS_FOLDER = "datasets"
OUTPUT_FOLDER = "data_plots"

dataset = "N14Bosch_1h_20190522134810.csv"

file_path = os.path.join(DATASETS_FOLDER, dataset)
series = read_csv(file_path, header=0, index_col=0, parse_dates=[0], infer_datetime_format=True)

def init():
    output_name = "boxplot_speed_diff_2019_04_dataset"

    one_series = series["speed_diff"]

    # one_year = one_series["2019"]
    # groups = one_year.groupby(Grouper(freq="M"))
    # months = concat([DataFrame(x[1].values) for x in groups], axis=1)
    # months = DataFrame(months)
    # months.columns = range(1, 13)
    # groups.plot()

    one_month = one_series["2019-04"]
    groups = one_month.groupby(Grouper(freq="D"))
    days = concat([DataFrame(x[1].values) for x in groups], axis=1)
    days = DataFrame(days)
    days.columns = range(1, 31)
    pyplot.gcf().set_size_inches(12, 7)
    days.boxplot()

    try:
        pyplot.savefig(os.path.join(OUTPUT_FOLDER, f"{output_name}.png"), format="png", dpi=300)
    except FileNotFoundError:
        os.makedirs(OUTPUT_FOLDER)
        pyplot.savefig(os.path.join(OUTPUT_FOLDER, f"{output_name}.png"), format="png", dpi=300)
