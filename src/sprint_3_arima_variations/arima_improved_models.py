import os
import sys
import shutil
import numpy
import time

from datetime import datetime
from math import sqrt
from pandas import read_csv, DataFrame
from matplotlib import pyplot
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error

ROOT_PATH = os.path.join(os.path.dirname(os.path.abspath(__name__)))
PATH = os.path.join(ROOT_PATH, "src", "sprint_3_arima_variations")

"""Adds the root folder to the sys path"""
sys.path.append(ROOT_PATH)

"""Sets the current folder as the current working directory"""
os.chdir(PATH)

from src.utils.csv_writer import CSVWriter

timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
OUTPUT_FOLDER = f"{timestamp}_results_arima"
results = list()
logs = list()


# Classes
class ArimaImprovedModel:
    """Class that represents the structure of an ARIMA improved model

    Author: Luis Marques
    """

    def __init__(self, series: DataFrame, arima_parameters: tuple, title: str = "", data_split: int = 0,
                 num_predictions: int = 10, predictions_size: float = 0.0):
        """Creates an instance of an ArimaImprovedModel.

        Args:
            series (DataFrame): series of the dataset to run the model.
            arima_parameters (tuple): parameters of the arima model.
            title (str): title of the model. Used to differentiate this model from other ones with the same parameters.
                Defaults to "".
            data_split (int): split number of the dataset used in the model. Defaults to 0.
            num_predictions (int): number of predictions of the model. Defaults to 10. It will only have effect if the
                predictions_size is equal to zero.
            predictions_size (float): percentage of data to predict (from 0 to 1). Defaults to 0.
        """

        self.series = series
        self.arima_parameters = arima_parameters
        self.title = title
        self.data_split = data_split
        self.values = self.series.values
        if predictions_size == 0.0:
            self.num_predictions = num_predictions
        else:
            self.num_predictions = int(len(self.values) * predictions_size)
        self.train = self.values[:-self.num_predictions]
        self.test = self.values[-self.num_predictions:]
        self.history = [x for x in self.train]
        self.predictions = list()
        self._set_name()
        self._set_folder()
        self._set_raw_file()
        self._execute()

    def _execute(self):
        """Executes the model"""
        self.starting_time = time.time()
        try:
            for timestep in range(self.num_predictions):
                model = ARIMA(self.history, order=self.arima_parameters)
                model_fit = model.fit()
                output = model_fit.forecast()
                prediction = output[0]
                self.predictions.append(prediction)
                obs = self.test[timestep]
                self.history.append(obs)
                self.file.write_line((str(prediction), str(obs)))

            self._export_plot()

        except Exception as err:
            logs.append(f"LOG: Model {self.name} exported with an error! {type(err).__name__}: {err}")
            self.execution_time = -1
            self.mae = -1
            self.mse = -1
            self.rmse = -1
            self.file.close()
            # If it returns an error the model folder is removed
            shutil.rmtree(self.folder)

        else:
            logs.append(f"LOG: Model {self.name} exported with success.")
            self.execution_time = time.time() - self.starting_time
            self.mae = mean_absolute_error(self.test, self.predictions)
            self.mse = mean_squared_error(self.test, self.predictions)
            self.rmse = sqrt(self.mse)
            self.file.close()

        finally:
            results.append((f'"{self.name}"', str(self.execution_time), str(self.mae), str(self.mse), str(self.rmse)))
            print(f"Model {self.name} finished.")

    def _set_name(self):
        """Sets the name of the model according to its variables"""
        self.name = ""
        self.title = "".join(self.title.split())
        if self.title != "":
            self.name += f"{self.title}_"
        self.name += "arima("
        for parameter in self.arima_parameters:
            self.name += f"{str(parameter)},"
        self.name = self.name[:-1] + ")_"
        self.name += f"predictions_{str(self.num_predictions)}"
        if self.data_split != 0:
            self.name += f"_crossvalidation_{self.data_split}"

    def _set_folder(self):
        """Creates an output folder for the model"""
        self.folder = os.path.join(OUTPUT_FOLDER, self.name)
        try:
            os.makedirs(self.folder)
        except FileNotFoundError:
            shutil.rmtree(self.folder)
            os.makedirs(self.folder)

    def _set_raw_file(self):
        """Creates a raw .csv file to write the predictions of the model"""
        file_name = f"raw_{self.name}.csv"
        file_path = os.path.join(self.folder, file_name)
        self.file = CSVWriter(file_path, ("Predict", self.series.name))

    # FIXME: Rever caso de quando o tamanho do self.values é menor que 50
    def _export_plot(self):
        """Exports the plot of the model"""
        timesteps = numpy.arange(len(self.values))
        train_size = 0
        train_values = tuple()

        if (len(self.values) <= 50):
            train_values = tuple([x for x in self.values[:-self.num_predictions]])
        elif (self.num_predictions < 50):
            timesteps = timesteps[-50:]
            train_size = 50 - self.num_predictions
            train_values = tuple([x for x in self.train[-train_size:]])
        else:
            timesteps = timesteps[-self.num_predictions:]

        real_values = (*[None for i in train_values[:-1]], *[train_values[len(train_values) - 1]],
                       *[x for x in self.test])

        prediction_values = (*[None for i in train_values], *[x for x in self.predictions])

        pyplot.plot(timesteps, real_values, color="green", marker="^", label="Real values")
        pyplot.plot(timesteps, prediction_values, color="red", marker="X", label="Predictions")
        pyplot.plot(timesteps[:train_size], train_values, color="blue", marker="o", label="Train values")

        pyplot.ylabel(self.series.name)
        pyplot.xlabel("Timesteps")
        pyplot.xticks(numpy.arange(min(timesteps), max(timesteps) + 1, 2.0))
        pyplot.grid(which="major", alpha=0.5)
        pyplot.gcf().canvas.set_window_title(self.name)
        pyplot.gcf().set_size_inches(15, 9)
        pyplot.savefig(os.path.join(self.folder, f"plot_{self.name}.png"), format="png", dpi=300)
        pyplot.close()


class ArimaMultivariateImprovedModel(ArimaImprovedModel):
    def __init__(self, series: DataFrame, arima_parameters: tuple, title: str = "", data_split: int = 0,
                 num_predictions: int = 10, predictions_size: float = 0.0):
        super().__init__(series, arima_parameters, title, data_split, num_predictions, predictions_size)


class SarimaImprovedModel(ArimaImprovedModel):
    def __init__(self, series: DataFrame, arima_parameters: tuple, title: str = "", data_split: int = 0,
                 num_predictions: int = 10, predictions_size: float = 0.0):
        super().__init__(series, arima_parameters, title, data_split, num_predictions, predictions_size)


class SarimaMultivariateImprovedModel(ArimaImprovedModel):
    def __init__(self, series: DataFrame, arima_parameters: tuple, title: str = "", data_split: int = 0,
                 num_predictions: int = 10, predictions_size: float = 0.0):
        super().__init__(series, arima_parameters, title, data_split, num_predictions, predictions_size)


# Functions
def init():
    dataset = "shampoo-sales.csv"
    # dataset = "daily-births.csv"

    def parser(x: int):
        return datetime.strptime(f"190{x}", "%Y-%m")

    # arima_parameters = list()
    # for p in range(1, 6):
    #     for d in range(0, 4):
    #         for q in range(0, 4):
    #             arima_parameters.append((p, d, q))

    arima_parameters = [(1, 2, 3), (2, 2, 3), (3, 2, 3), (4, 2, 3)]

    models = [
        {
            "model": ArimaImprovedModel,
            "arima_parameters": arima_parameters
        },
        # {
        #     "model": ArimaImprovedModel,
        #     "arima_parameters": arima_parameters
        # }
    ]

    num_predictions = 12

    title = "Daily Births"

    num_splits = 3

    results_order = "mse"

    run_models(dataset_name=dataset, date_parser=parser, models=models, num_predictions=num_predictions, title=title,
               num_splits=num_splits, results_order=results_order)
    # run_models(dataset_name=dataset, models=models, num_predictions=num_predictions, title=title,
    #            num_splits=num_splits, results_order=results_order)


def run_models(dataset_name: str, models: list, title: str, num_splits: int, results_order: str, num_predictions: int,
               predictions_size: float = 0, date_parser=None):
    """Parses the dataset (.csv file) into a DataFrame object and runs ARIMA models with the given dataset.

    Args:
        dataset_name (str): name of the .csv file with the dataset.
        models (list): list of dictionaries with the ARIMA models to be tested
        title (str): title to be used in the output files to distinguish the models
        num_splits (int): number of splits in case of being cross validation models
        results_order (str): order factor of the results list. ("name", "time", "mae", "mse" or "rmse").
        num_predictions (int): number of predictions of the model. Defaults to 10. It will only have effect if the
            predictions_size is equal to zero.
        predictions_size (float): percentage of data to predict (from 0 to 1). Defaults to 0.
        date_parser (optional): function to parse the dates of the dataset if needed. The function should return a
            datetime.
    """
    series = _dataset_to_series(dataset_name, date_parser)

    for model in models:
        for arima_parameters in model.get("arima_parameters"):
            model.get("model")(series=series, arima_parameters=arima_parameters, num_predictions=num_predictions,
                               predictions_size=predictions_size, title=title, data_split=num_splits)

    _export_results(results_order)
    _export_logs()


def _dataset_to_series(filename: str, date_parser=None):
    """Searches FILES_FOLDER for a dataset and returns it into a DataFrame object. If it is needed to parse the dates,
    a function should be passed as the "date_parser" argument.

    Args:
        filename (str): name of the .csv file containing the dataset. This file should be in the FILES_FOLDER.
        date_parser (optional): function to parse the dates of the dataset if needed. The function should return a
            datetime.
    """
    FILES_FOLDER = "files"
    file_path = os.path.join(FILES_FOLDER, filename)
    series = DataFrame()
    try:
        series = read_csv(file_path, header=0, index_col=0, parse_dates=False, squeeze=True, date_parser=date_parser)
    except Exception as err:
        print(f"('{filename}') {type(err).__name__}: {err}")
    return series


def _export_results(results_order: str = "name"):
    """Exports the results list into a .csv file ordered by the model order

    Args:
        results_order (str): order factor of the results list. ("name", "time", "mae", "mse" or "rmse"). Defaults to
            "name".
    """
    order = results_order.lower()
    if order == "time":
        order_num = 1
        order_name = "Execution Time (sec)"
    elif order == "mae":
        order_num = 2
        order_name = "MAE"
    elif order == "mse":
        order_num = 3
        order_name = "MSE"
    elif order == "rmse":
        order_num = 2
        order_name = "RMSE"
    else:
        order_num = 0
        order_name = "Model Name"

    results.sort(key=lambda line: float(line[order_num]))

    results_file = CSVWriter(os.path.join(OUTPUT_FOLDER, "results_summary.csv"), ("Model", "Execution Time (sec)",
                                                                                  "MAE", "MSE", "RMSE"))
    results_file.write_at_once(results)
    results_file.close()

    print(f"Results list file finished. Ordered by {order_name}.")


def _export_logs():
    """Exports the log list into a .txt file"""
    log_file = open(os.path.join(OUTPUT_FOLDER, "log.txt"), "w")
    for log in logs:
        log_file.write(log)
        log_file.write("\n")
    log_file.close()
    print("Log file finished.")
