import os
import shutil
import time
import sys

from pandas import read_csv, DataFrame
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime
from math import sqrt

ROOT_PATH = os.path.join(os.path.dirname(os.path.abspath(__name__)))
PATH = os.path.join(ROOT_PATH, "src", "sprint_2_arima_automation")

"""Adds the root folder to the sys path"""
sys.path.append(ROOT_PATH)

"""Sets the current folder as the current working directory"""
os.chdir(PATH)

from src.utils.csv_writer import CSVWriter

OUTPUT_FOLDER = "output"
ratings_list = list()
log_list = list()


class ArimaModel:
    """Class that represents the structure of my automated ARIMA model"""

    """Percentage os the train dataset"""
    TRAIN_SIZE = 0.66

    starting_time = time.time()

    """Resets the output folder"""
    try:
        shutil.rmtree(OUTPUT_FOLDER)
    except FileNotFoundError:
        pass

    def __init__(self, filename: str, arima_parameters: tuple, date_parser=None):
        """Creates an instance of an ARIMA model

        Args:
            filename (string): name of the file to create the model
            date_parser (function): function to parse the date
            p (int): number of lagged observations
            d (int): number of times that the raw observations are differenced
            q (int): size of the moving average window
        """

        self.arima_parameters = arima_parameters
        self.set_model_name()
        self.series = set_series(filename, date_parser)
        self.values = self.series.values
        self.train_size = int(len(self.values) * self.TRAIN_SIZE)
        self.train = self.values[0: self.train_size]
        self.test = self.values[self.train_size: len(self.values)]
        self.history = [x for x in self.train]
        self.predictions = list()
        self.set_output_folder()
        self.set_csv_file(["Predict", self.series.name])
        self.execute()

    def execute(self):
        """Executes the model"""
        try:
            for t in range(len(self.test)):
                model = ARIMA(self.history, order=self.arima_parameters)
                model_fit = model.fit(disp=0)
                output = model_fit.forecast()
                prediction = output[0][0]
                self.predictions.append(prediction)
                obs = self.test[t]
                self.history.append(obs)
                self.file.write_line((str(prediction), str(obs)))

            self.export_plot()

        except Exception as err:
            log_list.append(f">> Model {self.name} exported with an error! {type(err).__name__}: {err}")
            self.execution_time = -1
            self.mae = -1
            self.mse = -1
            self.rmse = -1
            self.file.close()
            # If it returns an error the model folder is removed
            shutil.rmtree(self.folder)

        else:
            log_list.append(f">> Model {self.name} exported with success.")
            self.execution_time = time.time() - self.starting_time
            self.mae = mean_absolute_error(self.test, self.predictions)
            self.mse = mean_squared_error(self.test, self.predictions)
            self.rmse = sqrt(self.mse)
            self.file.close()

        finally:
            ratings_list.append(
                (f'"{self.name}"', str(self.execution_time), str(self.mae), str(self.mse), str(self.rmse)))
            print(f"Model {self.name} finished.")

    def set_model_name(self):
        """Sets the name of the model"""
        self.name = "ARIMA("
        for index in self.arima_parameters:
            self.name += f"{str(index)},"
        self.name = self.name[:-1]
        self.name += ")"

    def set_output_folder(self):
        """Creates and sets an output folder for the model"""
        self.folder = os.path.join(OUTPUT_FOLDER, self.name)
        try:
            os.mkdir(self.folder)
        except FileNotFoundError:
            os.mkdir(OUTPUT_FOLDER)
            os.mkdir(self.folder)
        except FileExistsError:
            shutil.rmtree(self.folder)
            os.mkdir(self.folder)

    def set_csv_file(self, header: list):
        """Creates sets a writing csv file for the model"""
        file_name = f"{self.name}.csv"
        file_path = os.path.join(self.folder, file_name)
        self.file = CSVWriter(file_path, header)

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
        series = read_csv(file_path, header=0, index_col=0, parse_dates=False, squeeze=True, date_parser=date_parser)
    except FileNotFoundError as err:
        print(f"File Not Found ('{filename}'): {err}")
    return series


def run_arima_model(filename: str, arima_parameters_list: list, date_parser=None):
    """Runs ARIMA model

    Args:
        filename (str): name of the dataset file to import to the model.
        arima_parameters_list (list): list of all the tuples of parameters of the ARIMA model.
        date_parser (function, optional): function to parse the date. Defaults to None.
    """
    for arima_parameters in arima_parameters_list:
        ArimaModel(filename, arima_parameters, date_parser)


# 1 - Execution time (sec)
# 2 - MAE
# 3 - MSE
# 4 - RMSE
def export_ratings_list(order: int = 0):
    """Exports the ratings list into a .csv file

    Args:
        order (int, optional): Order factor of the ratings list
                               (1. Execution time (sec) / 2. MAE / 3. MSE / 4. RMSE). Defaults to 0.
    """
    ratings_list.sort(key=lambda line: float(line[order]))
    ratings_file = CSVWriter(os.path.join(OUTPUT_FOLDER, "model_ratings.csv"),
                              ["Model", "Execution Time (sec)", "MAE", "MSE", "RMSE"])
    ratings_file.write_at_once(ratings_list)
    ratings_file.close()
    if order == 1:
        print("Ratings list file finished. Ordered by Execution Time (sec).")
    elif order == 2:
        print("Ratings list file finished. Ordered by MAE.")
    elif order == 3:
        print("Ratings list file finished. Ordered by MSE.")
    elif order == 4:
        print("Ratings list file finished. Ordered by RMSE.")
    else:
        print("Ratings list file finished. Ordered by the model name.")


def export_log_file():
    """Exports the log list into a .txt file"""
    log_file = open(os.path.join(OUTPUT_FOLDER, "log.txt"), "w")
    for line in log_list:
        log_file.write(line)
        log_file.write("\n")
    log_file.close()
    print("Log file finshed.")


def init():
    """Main function"""

    def parser(x: int):
        """Parses the dates from shampoo-sales.csv dataset

        Args:
            x (int): last digit of the year

        Returns:
            datetime: new date time parsed from a string
        """
        return datetime.strptime(f"190{x}", "%Y-%m")

    arima_parameters_list = list()
    for p in range(1, 6):
        for d in range(0, 4):
            for q in range(0, 4):
                arima_parameters_list.append((p, d, q))

    run_arima_model(filename="shampoo-sales.csv", arima_parameters_list=arima_parameters_list, date_parser=parser)
    export_ratings_list(2)
    export_log_file()
