import os
import sys
import shutil
import numpy
import time

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


class ArimaImprovedModel:
    OUTPUT_FOLDER = "results_arima"
    results_list = list()
    log_list = list()

    def __init__(self, series: DataFrame, arima_parameters: tuple, num_predictions: int = 10, predictions_size: float = 0,
                 title: str = "", data_split: int = 0, order: str = "name"):
        """Creates an instance of an ArimaImprovedModel.

        Args:
            series (DataFrame): series of the dataset to run the model.
            arima_parameters (tuple): parameters of the arima model.
            num_predictions (int): number of predictions of the model. Defaults to 10. It will only have effect if the
                predictions_size is equal to zero.
            predictions_size (float): percentage of data to predict (from 0 to 1). Defaults to 0.
            title (str): title of the model. Used to defferentiate this model from other ones with the same parameters.
                Defaults to "".
            data_split (int): split number of the dataset used in the model. Defaults to 0.
            order (str): order factor of the results list. ("name", "time", "mae", "mse" or "rmse"). Defaults to "name".
        """

        self.series = series
        self.arima_parameters = arima_parameters
        self.values = self.series.values
        if predictions_size == 0:
            self.num_predictions = num_predictions
        else:
            self.num_predictions = int(len(self.values) * predictions_size)
        self.train = self.values[:-self.num_predictions]
        self.test = self.values[-self.num_predictions:]
        self.history = [x for x in self.train]
        self.predictions = list()
        self.title = title
        self.data_split = data_split
        self.order = order
        self._set_name()
        self._set_folder()
        self._set_raw_file()

    def _execute(self):
        """Executes the model"""
        self.starting_time = time.time()
        try:
            for timestep in range(self.num_predictions):
                model = ARIMA(self.history, order=self.arima_parameters)
                model_fit = model.fit(disp=0)
                output = model_fit.forecast()
                prediction = output[0][0]
                self.predictions.append(prediction)
                obs = self.test[timestep]
                self.history.append(obs)
                self.file.write_line((str(prediction), str(obs)))

            self._export_plot()

        except Exception as err:
            self.log_list.append(f"LOG: Model {self.name} exported with an error! {type(err).__name__}: {err}")
            self.execution_time = -1
            self.mae = -1
            self.mse = -1
            self.rmse = -1
            self.file.close()
            # If it returns an error the model folder is removed
            shutil.rmtree(self.folder)

        else:
            self.log_list.append(f"LOG: Model {self.name} exported with success.")
            self.execution_time = time.time() - self.starting_time
            self.mae = mean_absolute_error(self.test, self.predictions)
            self.mse = mean_squared_error(self.test, self.predictions)
            self.rmse = sqrt(self.mse)
            self.file.close()

        finally:
            self.results_list.append((f'"{self.name}"', str(self.execution_time), str(self.mae), str(self.mse), str(self.rmse)))
            print(f"Model {self.name} finished.")

    def _set_name(self):
        """Sets the name of the model according to its variables"""
        self.name = ""
        self.title = "".join(self.title.split())
        if not self.title:
            self.name += f"{self.title}_"
        self.name += "arima("
        for parameter in self.arima_parameters:
            self.name += f"{str(parameter)},"
        self.name = self.name[:-1] + ")_"
        self.name += f"predictions_{str(self.num_predictions)}"
        if self.data_split != 0:
            self.name += f"_crossvalidation_{self.data_split}"

    # FIXME: Se a pasta ja existir deve acrescentar um numero [ex:(2)] identificativo a OUTPUT_FOLDER
    def _set_folder(self):
        """Creates an output folder for the model"""
        self.folder = os.path.join(self.OUTPUT_FOLDER, self.name)
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

    def _export_plot(self):
        """Exports the plot of the model"""
        timesteps = numpy.arange(len(self.values))

        real_values_series = (*[None for i in self.train[:-1]], *[self.train[len(self.train) - 1]], *[x for x in self.test])

        prediction_values_series = (*[None for i in self.train], *[x for x in self.predictions])

        pyplot.plot(timesteps, real_values_series, color="green", marker="^", lineStyles="-", label="Real values")
        pyplot.plot(timesteps, prediction_values_series, color="red", marker="X", lineStyles="-", label="Predictions")
        pyplot.plot(numpy.arange(len(self.train)), self.train, color="blue", marker="o", lineStyles="-", label="Train values")

        pyplot.ylabel(self.series.name)
        pyplot.xlabel("Timesteps")
        pyplot.xticks(numpy.arange(min(timesteps), max(timesteps) + 1, 1.0))
        pyplot.grid(which="major", alpha=0.5)
        pyplot.gcf().canvas.set_window_title(self.name)
        pyplot.gcf().set_size_inches(12, 7)
        pyplot.savefig(os.path.join(self.folder, f"plot_{self.name}.png"), format="png", dpi=300)
        pyplot.close()

    def _export_results_list(self):
        """Exports the results list into a .csv file ordered by the model order"""
        order = self.order.lower()
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

        self.results_list.sort(key=lambda line: float(line[order_num]))

        results_file = CSVWriter(os.path.join(self.OUTPUT_FOLDER, "results_summary.csv"), ("Model", "Execution Time (sec)",
                                                                                           "MAE", "MSE", "RMSE"))
        results_file.write_at_once(self.results_list)
        results_file.close()

        print(f"Results list file finished. Ordered by {order_name}.")

    def _export_log_file(self):
        """Exports the log list into a .txt file"""
        log_file = open(os.path.join(self.OUTPUT_FOLDER, "log.txt"), "w")
        for line in self.log_list:
            log_file.write(line)
            log_file.write("\n")
        log_file.close()
        print("Log file finished.")


# class ArimaMultivariateImprovedModel(ArimaImprovedModel):
# class SarimaImprovedModel(ArimaImprovedModel):
# class SarimaMultivariateImprovedModel(ArimaImprovedModel):


def dataset_to_series(filename: str, date_parser: datetime = None):
    file_path = os.path.join("files", filename)
    series = DataFrame()
    try:
        series = read_csv(file_path, header=0, index_col=0, parse_dates=False, squeeze=True, date_parser=date_parser)
    except Exception as err:
        print(f"('{filename}') {type(err).__name__}: {err}")
    return series


# def run_models(dataset: str, models: tuple, arima_parameters: tuple, num_predictions: int, predictions_size: float, title: str,
#                order: str):
