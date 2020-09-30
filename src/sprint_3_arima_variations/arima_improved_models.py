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
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

# from sklearn.model_selection import TimeSeriesSplit

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

    # FIXME: Utilizar a classe TimeSeriesSplit para separar o dataset com os splits necessarios
    def __init__(self, series: DataFrame, variable_to_predict: str, arima_parameters: tuple, title: str = "",
                 data_split: int = 0, num_predictions: int = 10, predictions_size: float = 0.0):
        """Creates an instance of an ArimaImprovedModel.

        Args:
            series (DataFrame): series of the dataset to run the model.
            variable_to_predict (str): name of the variable to predict. It must be the same name of the column in the
                dataset.
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
        self.variable_to_predict = variable_to_predict
        self._set_values()
        if predictions_size == 0.0:
            self.num_predictions = num_predictions
        else:
            self.num_predictions = int(len(self.values) * predictions_size)
        self.train = self.values[:-self.num_predictions]
        self.test = self.values[-self.num_predictions:]
        self.history = [x for x in self.train]
        self._set_name()
        self._set_folder()
        self._set_raw_file()
        self._execute()

    def _execute(self):
        """Executes the model"""
        self.starting_time = time.time()
        try:
            # Fazer as previsoes todas de uma vez
            # model = ARIMA(endog=self.history, order=self.arima_parameters)
            # model_fit = model.fit()
            # predictions = model_fit.forecast(steps=self.num_predictions)

            # Fazer uma previsao de cada vez
            predictions = list()
            for timestep in range(self.num_predictions):
                model = ARIMA(self.history, order=self.arima_parameters)
                model_fit = model.fit()
                output = model_fit.forecast()
                prediction = output[0]
                predictions.append(prediction)
                obs = self.test[timestep]
                self.history.append(obs)

            self.predictions = self.scaler.inverse_transform([predictions])[0]
            self.train = self.scaler.inverse_transform([self.train])[0]
            self.test = self.scaler.inverse_transform([self.test])[0]

            for timestep in range(self.num_predictions):
                self.file.write_line((str(self.predictions[timestep]), str(self.test[timestep])))

            self._export_plot()
            self.file.close()

        except Exception as err:
            logs.append(f"LOG: Model {self.name} exported with an error! {type(err).__name__}: {err}")
            self.execution_time = -1
            self.mae = -1
            self.mse = -1
            self.rmse = -1
            # If it returns an error the model folder is removed
            shutil.rmtree(self.folder)

        else:
            logs.append(f"LOG: Model {self.name} exported with success.")
            self.execution_time = time.time() - self.starting_time
            self.mae = mean_absolute_error(self.test, self.predictions)
            self.mse = mean_squared_error(self.test, self.predictions)
            self.rmse = sqrt(self.mse)

        finally:
            results.append((f'"{self.name}"', str(self.execution_time), str(self.mae), str(self.mse), str(self.rmse)))
            print(f"Model {self.name} finished.")

    def _set_values(self):
        """Sets the values to be used based on the series and the variable to predict"""
        self.scaler = MinMaxScaler()
        self.values = self.series[[self.variable_to_predict]]
        self.values[[self.variable_to_predict]] = self.scaler.fit_transform(self.values[[self.variable_to_predict]])
        self.values = getattr(self.values, self.variable_to_predict).values

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
        self.file = CSVWriter(file_path, ("Predict", self.variable_to_predict))

    def _export_plot(self):
        """Exports the plot of the model"""
        timesteps = numpy.arange(len(self.values))
        train_size = 3
        train_values = tuple([x for x in self.train[-train_size:]])

        if len(timesteps) <= 50:
            train_size = len(self.train)
            train_values = tuple([x for x in self.train])
        elif self.num_predictions < 50:
            timesteps = timesteps[-50:]
            train_size = 50 - self.num_predictions
            train_values = tuple([x for x in self.train[-train_size:]])
        else:
            timesteps = timesteps[-self.num_predictions - train_size:]

        real_values = (*[None for x in train_values[:-1]], train_values[-1], *[x for x in self.test])

        prediction_values = (*[None for x in train_values], *[x for x in self.predictions])

        pyplot.plot(timesteps, real_values, color="green", marker="^", label="Real values")
        pyplot.plot(timesteps, prediction_values, color="red", marker="X", label="Predictions")
        pyplot.plot(timesteps[:train_size], train_values, color="blue", marker="o", label="Train values")

        x_interval = 1.0
        if timesteps[-1] > 99:
            x_interval = 2.0
        elif timesteps[-1] > 999:
            x_interval = 3.0
        elif timesteps[-1] > 9999:
            x_interval = 4.0
        elif timesteps[-1] > 99999:
            x_interval = 5.0

        pyplot.ylabel(self.variable_to_predict)
        pyplot.xlabel("Timesteps")
        pyplot.xticks(numpy.arange(min(timesteps), max(timesteps) + 1, x_interval))
        pyplot.grid(which="major", alpha=0.5)
        pyplot.gcf().canvas.set_window_title(self.name)
        pyplot.gcf().set_size_inches(15, 9)
        pyplot.savefig(os.path.join(self.folder, f"plot_{self.name}.png"), format="png", dpi=300)
        pyplot.close()


class ArimaMultivariateImprovedModel(ArimaImprovedModel):
    """Class that represents the structure of an ARIMA multivariate improved model

    Author: Luis Marques
    """

    def __init__(self, series: DataFrame, variable_to_predict: str, exog_variables: tuple, arima_parameters: tuple,
                 title: str = "", data_split: int = 0, num_predictions: int = 10, predictions_size: float = 0.0):
        """Creates an instance of an ArimaMultivariateImprovedModel.

        Args:
            series (DataFrame): series of the dataset to run the model.
            variable_to_predict (str): name of the variable to predict. It must be the same name of the column in the
                dataset.
            exog_variables (tuple): tuple of exogenous variables to help the model to predict the values.
            arima_parameters (tuple): parameters of the arima model.
            title (str): title of the model. Used to differentiate this model from other ones with the same parameters.
                Defaults to "".
            data_split (int): split number of the dataset used in the model. Defaults to 0.
            num_predictions (int): number of predictions of the model. Defaults to 10. It will only have effect if the
                predictions_size is equal to zero.
            predictions_size (float): percentage of data to predict (from 0 to 1). Defaults to 0.
        """
        self.exog_variables = exog_variables
        super().__init__(series, variable_to_predict, arima_parameters, title, data_split, num_predictions,
                         predictions_size)

    def _execute(self):
        """Executes the model"""
        self.starting_time = time.time()
        try:
            # # Fazer as previsoes todas de uma vez
            # history_extra = tuple([x for x in self.exog_values[:len(self.history)]])
            # test_extra = tuple([x for x in self.exog_values[-len(self.test):]])
            # model = ARIMA(endog=self.history, order=self.arima_parameters, exog=history_extra)
            # model_fit = model.fit()
            # predictions = model_fit.forecast(steps=self.num_predictions, exog=test_extra)

            # Fazer uma previsao de cada vez
            predictions = list()
            for timestep in range(self.num_predictions):
                history_extra = tuple([x for x in self.exog_values[:len(self.history)]])
                test_extra = tuple(self.exog_values[-len(self.test) + timestep])
                model = ARIMA(self.history, order=self.arima_parameters, exog=history_extra)
                model_fit = model.fit()
                output = model_fit.forecast(exog=test_extra)
                prediction = output[0]
                predictions.append(prediction)
                obs = self.test[timestep]
                self.history.append(obs)

            self.predictions = self.scaler.inverse_transform([predictions])[0]
            self.train = self.scaler.inverse_transform([self.train])[0]
            self.test = self.scaler.inverse_transform([self.test])[0]

            for timestep in range(self.num_predictions):
                self.file.write_line((str(self.predictions[timestep]), str(self.test[timestep])))

            self._export_plot()
            self.file.close()

        except Exception as err:
            logs.append(f"LOG: Model {self.name} exported with an error! {type(err).__name__}: {err}")
            self.execution_time = -1
            self.mae = -1
            self.mse = -1
            self.rmse = -1
            # If it returns an error the model folder is removed
            shutil.rmtree(self.folder)

        else:
            logs.append(f"LOG: Model {self.name} exported with success.")
            self.execution_time = time.time() - self.starting_time
            self.mae = mean_absolute_error(self.test, self.predictions)
            self.mse = mean_squared_error(self.test, self.predictions)
            self.rmse = sqrt(self.mse)

        finally:
            results.append((f'"{self.name}"', str(self.execution_time), str(self.mae), str(self.mse), str(self.rmse)))
            print(f"Model {self.name} finished.")

    def _set_values(self):
        """Sets the values arrays to be used based on the series and the variable to predict. Also sets the exog
        values to be used in the predictions."""
        self.exog_values = self.series.filter(items=self.exog_variables)
        self.exog_values = self.exog_values.values
        super()._set_values()

    def _set_name(self):
        """Sets the name of the model according to its variables"""
        self.name = ""
        self.title = "".join(self.title.split())
        if self.title != "":
            self.name += f"{self.title}_"
        self.name += "arimax("
        for parameter in self.arima_parameters:
            self.name += f"{str(parameter)},"
        self.name = self.name[:-1] + ")_"
        self.name += f"predictions_{str(self.num_predictions)}"
        if self.data_split != 0:
            self.name += f"_crossvalidation_{self.data_split}"


class SarimaImprovedModel(ArimaImprovedModel):
    """Class that represents the structure of a SARIMA improved model

    Author: Luis Marques
    """

    def __init__(self, series: DataFrame, variable_to_predict: str, arima_parameters: tuple, season_parameters: tuple,
                 title: str = "", data_split: int = 0, num_predictions: int = 10, predictions_size: float = 0.0):
        """Creates an instance of an SarimaImprovedModel.

        Args:
            series (DataFrame): series of the dataset to run the model.
            variable_to_predict (str): name of the variable to predict. It must be the same name of the column in the
                dataset.
            arima_parameters (tuple): parameters of the sarima model.
            season_parameters (tuple): season parameters of the sarima model.
            title (str): title of the model. Used to differentiate this model from other ones with the same parameters.
                Defaults to "".
            data_split (int): split number of the dataset used in the model. Defaults to 0.
            num_predictions (int): number of predictions of the model. Defaults to 10. It will only have effect if the
                predictions_size is equal to zero.
            predictions_size (float): percentage of data to predict (from 0 to 1). Defaults to 0.
        """
        self.season_parameters = season_parameters
        super().__init__(series, variable_to_predict, arima_parameters, title, data_split, num_predictions,
                         predictions_size)

    def _execute(self):
        """Executes the model"""
        self.starting_time = time.time()
        try:
            # Fazer as previsoes todas de uma vez
            # model = ARIMA(self.history, order=self.arima_parameters)
            # model_fit = model.fit()
            # predictions = model_fit.forecast(steps=self.num_predictions)

            # Fazer uma previsao de cada vez
            predictions = list()
            for timestep in range(self.num_predictions):
                model = SARIMAX(self.history, order=self.arima_parameters, seasonal_order=self.season_parameters,
                                enforce_stationarity=False, enforce_invertibility=False)
                model_fit = model.fit()
                output = model_fit.forecast()
                prediction = output[0]
                predictions.append(prediction)
                obs = self.test[timestep]
                self.history.append(obs)

            self.predictions = self.scaler.inverse_transform([predictions])[0]
            self.train = self.scaler.inverse_transform([self.train])[0]
            self.test = self.scaler.inverse_transform([self.test])[0]

            for timestep in range(self.num_predictions):
                self.file.write_line((str(self.predictions[timestep]), str(self.test[timestep])))

            self._export_plot()
            self.file.close()

        except Exception as err:
            logs.append(f"LOG: Model {self.name} exported with an error! {type(err).__name__}: {err}")
            self.execution_time = -1
            self.mae = -1
            self.mse = -1
            self.rmse = -1
            # If it returns an error the model folder is removed
            shutil.rmtree(self.folder)

        else:
            logs.append(f"LOG: Model {self.name} exported with success.")
            self.execution_time = time.time() - self.starting_time
            self.mae = mean_absolute_error(self.test, self.predictions)
            self.mse = mean_squared_error(self.test, self.predictions)
            self.rmse = sqrt(self.mse)

        finally:
            results.append((f'"{self.name}"', str(self.execution_time), str(self.mae), str(self.mse), str(self.rmse)))
            print(f"Model {self.name} finished.")

    def _set_name(self):
        """Sets the name of the model according to its variables"""
        self.name = ""
        self.title = "".join(self.title.split())
        if self.title != "":
            self.name += f"{self.title}_"
        self.name += "sarima("
        for parameter in self.arima_parameters:
            self.name += f"{str(parameter)},"
        self.name = self.name[:-1] + ")("
        for parameter in self.season_parameters:
            self.name += f"{str(parameter)},"
        self.name = self.name[:-1] + ")_"
        self.name += f"predictions_{str(self.num_predictions)}"
        if self.data_split != 0:
            self.name += f"_crossvalidation_{self.data_split}"


class SarimaMultivariateImprovedModel(ArimaImprovedModel):
    """Class that represents the structure of a SARIMA multivariate improved model

    Author: Luis Marques
    """

    def __init__(self, series: DataFrame, variable_to_predict: str, exog_variables: tuple, arima_parameters: tuple,
                 season_parameters: tuple, title: str = "", data_split: int = 0, num_predictions: int = 10,
                 predictions_size: float = 0.0):
        """Creates an instance of an SarimaMultivariateImprovedModel.

        Args:
            series (DataFrame): series of the dataset to run the model.
            variable_to_predict (str): name of the variable to predict. It must be the same name of the column in the
                dataset.
            exog_variables (tuple): tuple of exogenous variables to help the model to predict the values.
            arima_parameters (tuple): parameters of the sarima model.
            season_parameters (tuple): season parameters of the sarima model.
            title (str): title of the model. Used to differentiate this model from other ones with the same parameters.
                Defaults to "".
            data_split (int): split number of the dataset used in the model. Defaults to 0.
            num_predictions (int): number of predictions of the model. Defaults to 10. It will only have effect if the
                predictions_size is equal to zero.
            predictions_size (float): percentage of data to predict (from 0 to 1). Defaults to 0.
        """
        self.exog_variables = exog_variables
        self.season_parameters = season_parameters
        super().__init__(series, variable_to_predict, arima_parameters, title, data_split, num_predictions,
                         predictions_size)

    def _execute(self):
        """Executes the model"""
        self.starting_time = time.time()
        try:
            # # Fazer as previsoes todas de uma vez
            # history_extra = tuple([x for x in self.exog_values[:len(self.history)]])
            # test_extra = tuple([x for x in self.exog_values[-len(self.test):]])
            # model = ARIMA(self.history, order=self.arima_parameters, exog=history_extra)
            # model_fit = model.fit()
            # predictions = model_fit.forecast(steps=self.num_predictions, exog=test_extra)

            # Fazer uma previsao de cada vez
            predictions = list()
            for timestep in range(self.num_predictions):
                history_extra = tuple([x for x in self.exog_values[:len(self.history)]])
                test_extra = tuple(self.exog_values[-len(self.test) + timestep])
                model = SARIMAX(self.history, exog=history_extra, order=self.arima_parameters,
                                seasonal_order=self.season_parameters, enforce_stationarity=False,
                                enforce_invertibility=False)
                model_fit = model.fit()
                output = model_fit.forecast(exog=test_extra)
                prediction = output[0]
                predictions.append(prediction)
                obs = self.test[timestep]
                self.history.append(obs)

            self.predictions = self.scaler.inverse_transform([predictions])[0]
            self.train = self.scaler.inverse_transform([self.train])[0]
            self.test = self.scaler.inverse_transform([self.test])[0]

            for timestep in range(self.num_predictions):
                self.file.write_line((str(self.predictions[timestep]), str(self.test[timestep])))

            self._export_plot()
            self.file.close()

        except Exception as err:
            logs.append(f"LOG: Model {self.name} exported with an error! {type(err).__name__}: {err}")
            self.execution_time = -1
            self.mae = -1
            self.mse = -1
            self.rmse = -1
            # If it returns an error the model folder is removed
            shutil.rmtree(self.folder)

        else:
            logs.append(f"LOG: Model {self.name} exported with success.")
            self.execution_time = time.time() - self.starting_time
            self.mae = mean_absolute_error(self.test, self.predictions)
            self.mse = mean_squared_error(self.test, self.predictions)
            self.rmse = sqrt(self.mse)

        finally:
            results.append((f'"{self.name}"', str(self.execution_time), str(self.mae), str(self.mse), str(self.rmse)))
            print(f"Model {self.name} finished.")

    def _set_values(self):
        """Sets the values arrays to be used based on the series and the variable to predict. Also sets the exog
        values to be used in the predictions."""
        self.exog_values = self.series.filter(items=self.exog_variables)
        self.exog_values = self.exog_values.values
        super()._set_values()

    def _set_name(self):
        """Sets the name of the model according to its variables"""
        self.name = ""
        self.title = "".join(self.title.split())
        if self.title != "":
            self.name += f"{self.title}_"
        self.name += "sarimax("
        for parameter in self.arima_parameters:
            self.name += f"{str(parameter)},"
        self.name = self.name[:-1] + ")("
        for parameter in self.season_parameters:
            self.name += f"{str(parameter)},"
        self.name = self.name[:-1] + ")_"
        self.name += f"predictions_{str(self.num_predictions)}"
        if self.data_split != 0:
            self.name += f"_crossvalidation_{self.data_split}"


# Functions
def init():
    # dataset = "shampoo-sales.csv"
    # dataset = "daily-births.csv"
    dataset = "N14Bosch_2019-04.csv"

    variable_to_predict = "speed_diff"

    # def parser(x: int):
    #     return datetime.strptime(f"190{x}", "%Y-%m")

    # arima_parameters = list()
    # for p in range(1, 6):
    #     for d in range(0, 4):
    #         for q in range(0, 4):
    #             arima_parameters.append((p, d, q))

    arima_parameters = [(2, 2, 3), (4, 2, 3)]

    models = [
        {
            "model": ArimaImprovedModel,
            "arima_parameters": arima_parameters
        },
        {
            "model": ArimaMultivariateImprovedModel,
            "arima_parameters": arima_parameters,
            "exog_variables": ("precipitation", "week_day")
        },
        # {
        #     "model": SarimaImprovedModel,
        #     "arima_parameters": arima_parameters,
        #     "season_parameters": [(0, 0, 1, 24), (0, 1, 1, 24)]
        # },
        # {
        #     "model": SarimaMultivariateImprovedModel,
        #     "arima_parameters": arima_parameters,
        #     "exog_variables": ("precipitation", "week_day"),
        #     "season_parameters": [(0, 0, 1, 24), (0, 1, 1, 24)]
        # }
    ]

    num_predictions = 15

    title = "Births test"

    num_splits = 3

    results_order = "mse"

    # run_models(dataset_name=dataset, variable_to_predict=variable_to_predict, date_parser=parser, models=models,
    #            num_predictions=num_predictions, title=title, num_splits=num_splits, results_order=results_order)
    run_models(dataset_name=dataset, variable_to_predict=variable_to_predict, models=models,
               num_predictions=num_predictions, title=title, num_splits=num_splits, results_order=results_order)


def run_models(dataset_name: str, models: list, variable_to_predict: str, title: str, num_splits: int,
               results_order: str, num_predictions: int, predictions_size: float = 0, date_parser=None):
    """Parses the dataset (.csv file) into a DataFrame object and runs ARIMA models with the given dataset.

    Args:
        dataset_name (str): name of the .csv file with the dataset.
        models (list): list of dictionaries with the ARIMA models to be tested.
        title (str): title to be used in the output files to distinguish the models.
        variable_to_predict (str): name of the variable to predict. It must be the same name of the column in the
            dataset.
        num_splits (int): number of splits in case of being cross validation models.
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
            if model.get("model") == ArimaImprovedModel:
                model.get("model")(series=series, variable_to_predict=variable_to_predict,
                                   arima_parameters=arima_parameters, num_predictions=num_predictions,
                                   predictions_size=predictions_size, title=title, data_split=num_splits)
            elif model.get("model") == ArimaMultivariateImprovedModel:
                exog_variables = model.get("exog_variables")
                model.get("model")(series=series, variable_to_predict=variable_to_predict,
                                   exog_variables=exog_variables, arima_parameters=arima_parameters,
                                   num_predictions=num_predictions, predictions_size=predictions_size, title=title,
                                   data_split=num_splits)
            elif model.get("model") == SarimaImprovedModel:
                for season_parameters in model.get("season_parameters"):
                    model.get("model")(series=series, variable_to_predict=variable_to_predict,
                                       arima_parameters=arima_parameters, season_parameters=season_parameters,
                                       num_predictions=num_predictions, predictions_size=predictions_size, title=title,
                                       data_split=num_splits)
            elif model.get("model") == SarimaMultivariateImprovedModel:
                exog_variables = model.get("exog_variables")
                for season_parameters in model.get("season_parameters"):
                    model.get("model")(series=series, variable_to_predict=variable_to_predict,
                                       exog_variables=exog_variables, arima_parameters=arima_parameters,
                                       season_parameters=season_parameters, num_predictions=num_predictions,
                                       predictions_size=predictions_size, title=title, data_split=num_splits)
            else:
                logs.append(f"LOG: Model {model.get('model')} was not found!")
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
        series = read_csv(file_path, header=0, index_col=0, parse_dates=[0], infer_datetime_format=True,
                          date_parser=date_parser)
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
