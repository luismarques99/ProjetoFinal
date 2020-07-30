import os
import shutil
import time
import sys
import numpy

from pandas import read_csv, DataFrame
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime
from math import sqrt

ROOT_PATH = os.path.join(os.path.dirname(os.path.abspath(__name__)))
PATH = os.path.join(ROOT_PATH, "src", "sprint_3_arima_variations")

"""Adds the root folder to the sys path"""
sys.path.append(ROOT_PATH)

"""Sets the current folder as the current working directory"""
os.chdir(PATH)

from src.utils.csv_writer import CSVWriter

OUTPUT_FOLDER = "output"
ratings_list = list()
log_list = list()


class ArimaImprovedModel:
    """Class that represents the structure of my automated ARIMA model"""

    """Resets the output folder"""
    try:
        shutil.rmtree(OUTPUT_FOLDER)
    except FileNotFoundError:
        pass

    def __init__(self, filename: str,
                 arima_parameters: tuple,
                 num_predictions: int = 0,
                 predictions_size: float = 0,
                 date_parser: datetime = None):
        """Creates an instance of an ARIMA model

        Args:
            filename (string): name of the file to create the model.
            arima_parameters (tuple): tuple with the parameters for the model.
            num_predictions (int, optional): number of predictions of the model. If num_predictions is different than
            zero, predictions_size is not needed. Defaults to 0.
            predictions_size (float, optional): percentage of the test dataset (from 0 to 1). Only needed when
            num_predictions is different than zero. Defaults to 0.
            date_parser (datetime, optional): function to parse the date. Defaults to None.
        """
        # FIXME: DUVIDA - é necessário instanciar as variaveis todas no inicio antes de utiliza-las?
        # FIXME: Exemplo: name, starting_time, execution_time, ...
        self.arima_parameters = arima_parameters
        self.name = str()
        self.series = set_series(filename, date_parser)
        self.values = self.series.values
        if num_predictions != 0:
            self.train_size = len(self.values) - num_predictions
        else:
            self.train_size = int(len(self.values) * (1 - predictions_size))
        self.num_predictions = len(self.values) - self.train_size
        self.train = self.values[0: self.train_size]
        self.test = self.values[self.train_size: len(self.values)]
        self.history = [x for x in self.train]
        self.predictions = list()
        self.starting_time = float()
        self.execution_time = float()
        self.mae = float()
        self.mse = float()
        self.rmse = float()
        self.folder = str()
        self.file = None
        self.set_model_name()
        self.set_output_folder()
        self.set_csv_file(("Predict", self.series.name))
        self.execute()

    def execute(self):
        """Executes the model"""
        self.starting_time = time.time()
        try:
            for timestep in range(self.num_predictions):
                model = ARIMA(self.history, order=self.arima_parameters)
                model_fit = model.fit(disp=0)
                # TODO: DUVIDA - Qual a diferença entre estes outputs e os que eu tenho na minha classe?
                output = model_fit.forecast()
                prediction = output[0][0]
                self.predictions.append(prediction)
                obs = self.test[timestep]
                self.history.append(obs)
                self.file.write_line((str(prediction), str(obs)))

            self.export_plot()

        except Exception as err:
            log_list.append(f">> Model {self.name} exported with an error! {type(err).__name__}: {err}")
            # print(f">> Model {self.name} exported with an error! {type(err).__name__}: {err}")
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
                (
                    f'"{self.name}"',
                    str(self.execution_time),
                    str(self.mae),
                    str(self.mse),
                    str(self.rmse)
                )
            )
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

    def set_csv_file(self, header: tuple):
        """Creates sets a writing csv file for the model"""
        file_name = f"{self.name}.csv"
        file_path = os.path.join(self.folder, file_name)
        self.file = CSVWriter(file_path, header)

    def export_plot(self):
        """Exports the plot to a folder"""
        timesteps = numpy.arange(len(self.values))

        real_values_series = (
            *[None for i in self.train[:-1]],
            *[self.train[len(self.train) - 1]],
            *[x for x in self.test]
        )

        prediction_values_series = (
            *[None for i in self.train],
            *[x for x in self.predictions]
        )

        pyplot.plot(timesteps,
                    real_values_series,
                    color="green",
                    marker="^",
                    linestyle="-",
                    label="Real values")
        pyplot.plot(timesteps,
                    prediction_values_series,
                    color="red",
                    marker="X",
                    linestyle="-",
                    label="Predictions")
        pyplot.plot(numpy.arange(len(self.train)),
                    self.train,
                    color="blue",
                    marker="o",
                    linestyle="-",
                    label="Train values")
        pyplot.ylabel(self.series.name)
        pyplot.xlabel("Timesteps")
        pyplot.xticks(numpy.arange(min(timesteps), max(timesteps) + 1, 1.0))
        pyplot.grid(which="major", alpha=0.5)
        pyplot.gcf().canvas.set_window_title(self.name)
        pyplot.gcf().set_size_inches(12, 7)
        pyplot.savefig(os.path.join(self.folder, f"{self.name}_plot.png"), format="png", dpi=300)
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
    return series.copy()


def run_arima_model(filename: str, arima_parameters_list: list, date_parser=None):
    """Runs ARIMA model

    Args:
        filename (str): name of the dataset file to import to the model.
        arima_parameters_list (list): list of all the tuples of parameters of the ARIMA model.
        date_parser (function, optional): function to parse the date. Defaults to None.
    """
    for arima_parameters in arima_parameters_list:
        ArimaImprovedModel(filename=filename,
                           arima_parameters=arima_parameters,
                           num_predictions=12,
                           predictions_size=0.34,
                           date_parser=date_parser)


def export_ratings_list(order: str):
    """Exports the ratings list into a .csv file ordered by the parameter chosen

    Args:
        order (str, optional): Order factor of the ratings list. ("execution time", "mae", "mse" or "rmse")
    """
    order = order.lower()
    if order == "execution time":
        order_num = 1
        order_name = "Execution Time (sec)"
    elif order == "mae":
        order_num = 2
        order_name = "MAE"
    elif order == "mse":
        order_num = 3
        order_name = "MSE"
    elif order == "rmse":
        order_num = 4
        order_name = "RMSE"
    else:
        order_num = 0
        order_name = "Model Name"

    ratings_list.sort(key=lambda line: float(line[order_num]))

    ratings_file = CSVWriter(os.path.join(OUTPUT_FOLDER, "model_ratings.csv"),
                             ("Model", "Execution Time (sec)", "MAE", "MSE", "RMSE"))
    ratings_file.write_at_once(ratings_list)
    ratings_file.close()

    print(f"Ratings list file finished. Ordered by {order_name}.")


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
    export_ratings_list("mse")
    export_log_file()
