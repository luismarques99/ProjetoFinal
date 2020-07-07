import os
import shutil
import time
import sys

# Adds the root folder to the sys path
ROOT_PATH = os.path.join(os.path.dirname(os.path.abspath(__name__)))
sys.path.append(ROOT_PATH)

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.arima_model import ARIMA
from pandas import DataFrame, concat, read_csv
from numpy import array, concatenate
from math import sqrt
from matplotlib import pyplot

from my_modules.csv_writer import csv_writer

PATH = os.path.join(".", "sprint-3_arima_variations")
os.chdir(PATH)

OUTPUT_FOLDER = "results_arimax_2"
ratings_list = list()
log_list = list()


class arima_multivariate_improved:
	"""Class that represents the structure of an arima multivariate improved model"""

	starting_time = time.time()

	def __init__(self, filename: str):
		...

	def create_folder(self):
        """Creates a folder for the model

        Returns:
            string: folder path
        """
        folder = os.path.join(OUTPUT_FOLDER, self.name)
        try:
            os.mkdir(folder)
        except FileNotFoundError:
            os.mkdir(OUTPUT_FOLDER)
            os.mkdir(folder)
        except FileExistsError:
            shutil.rmtree(folder)
            os.mkdir(folder)
        return folder

    def create_file(self, header: list()):
        """Creates and opens a writing file for the model

        Returns:
            file: file ready to write
        """
        file_name = f"{self.name}.csv"
        file_path = os.path.join(self.folder, file_name)
        file = csv_writer(file_path, header)
        return file

    def export_plot(self):
        """Exports the plot to a folder"""
        pyplot.plot(self.train, color="blue")
        pyplot.plot([None for i in self.train] + [x for x in self.test], color="green")
        pyplot.plot([None for i in self.train] + [x for x in self.predictions], color="red")
        pyplot.gcf().canvas.set_window_title(self.name)
        pyplot.savefig(os.path.join(self.folder, f"{self.name}-plot.png"))
        pyplot.close()


def set_series(filename: str, date_parser=None):
    """Set the series

    Args:
        filename (str): name of the file to read (the file must be inside the folder 'files').
        date_parser (function, optional): function to parse the date. Defaults to None.

    Returns:
        DataFrame: time series
    """
    file_path = os.path.join("files", filename)
    series = DataFrame()
    try:
        series = read_csv(file_path, header=0, index_col=0, parse_dates=False, squeeze=True, date_parser=date_parser,)
    except FileNotFoundError as err:
        print(f"File Not Found ('{filename}'): {err}")
    return series