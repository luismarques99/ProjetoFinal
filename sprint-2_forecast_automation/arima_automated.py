import os
import shutil

from pandas import read_csv, DataFrame, datetime
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

from csv_writer import csv_writer


PATH = os.path.join(".", "sprint-2_forecast_automation")
os.chdir(PATH)


OUTPUT_FOLDER = "output"
mse_list = list()
log_list = list()


class arima_model:
    """Class that represents the structure of my automated ARIMA model"""

    TRAIN_SIZE = 0.66

    """Resets the output folder"""
    try:
        shutil.rmtree(OUTPUT_FOLDER)
    except FileNotFoundError:
        pass

    def __init__(self, filename: str, date_parser, p: int, d: int, q: int):
        """Creates an instance of an ARIMA model

        Args:
            filename (string): name of the file to create the model
            date_parser (function): function to parse the date
            p (int): number of lagged observations
            d (int): number of times that the raw observations are differenced
            q (int): size of the moving average window
        """
        self.p = p
        self.d = d
        self.q = q
        self.series = set_series(filename, date_parser)
        self.name = f"ARIMA({self.p},{self.d},{self.q})"
        self.values = self.series.values
        self.train_size = int(len(self.values) * self.TRAIN_SIZE)
        self.train = self.values[0 : self.train_size]
        self.test = self.values[self.train_size : len(self.values)]
        self.history = [x for x in self.train]
        self.predictions = list()
        self.folder = self.create_folder()
        self.file = self.create_file(["Predict", self.series.name])
        self.execute()

    def execute(self):
        """Executes the model"""

        try:
            for t in range(len(self.test)):
                model = ARIMA(self.history, order=(self.p, self.d, self.q))
                model_fit = model.fit(disp=0)
                output = model_fit.forecast()
                prediction = output[0][0]
                self.predictions.append(prediction)
                obs = self.test[t]
                self.history.append(obs)
                self.file.write_line([str(prediction), str(obs)])

            self.mse = mean_squared_error(self.test, self.predictions)
            mse_list.append([str(self.mse), f'"{self.name}"'])
            mse_list.sort(key=lambda item: float(item[0]))
            self.export_plot()

        except Exception as err:
            log_list.append(f">> Model ARIMA({self.p}, {self.d} , {self.q}) not exported! {type(err).__name__}: {err}")
            self.file.close()
            shutil.rmtree(self.folder)
            pass

        else:
            log_list.append(f">> Model ARIMA({self.p}, {self.d} , {self.q}) exported with success.")
            pass

        finally:
            self.file.close()

    def create_folder(self):
        """Creates a folder for the model

        Returns:
            string: folder path
        """
        single_folder = f"ARIMA({self.p},{self.d},{self.q})"
        folder = os.path.join(OUTPUT_FOLDER, single_folder)
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
        file_name = f"ARIMA({self.p},{self.d},{self.q}).csv"
        file_path = os.path.join(self.folder, file_name)
        file = csv_writer(file_path, header)
        return file

    def export_plot(self):
        """Exports the plot to a folder"""
        pyplot.plot(self.train, color="blue")
        pyplot.plot([None for i in self.train] + [x for x in self.test], color="green")
        pyplot.plot([None for i in self.train] + [x for x in self.predictions], color="red")
        pyplot.gcf().canvas.set_window_title(f"ARIMA({self.p}, {self.d}, {self.q})")
        pyplot.savefig(os.path.join(self.folder, f"ARIMA({self.p},{self.d},{self.q})-plot.png"))
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


def arima_automated(
    filename: str, date_parser=None, p_range: list() = [1, 2], d_range: list() = [0, 1], q_range: list() = [0, 1],
):
    """ARIMA model automated

    Args:
        filename (str): name of the dataset file to import to the model.
        date_parser (function, optional): function to parse the date. Defaults to None.
        p_range (list, optional): range of values for the number of lagged observations. Defaults to [1, 2].
        d_range (list, optional): range of values for the number of times that the raw observations are differenced.
                                  Defaults to [0, 1].
        q_range (list, optional): range of values for the size of the moving average window. Defaults to [0, 1].
    """
    for p in list(range(p_range[0], p_range[-1])):
        for d in list(range(d_range[0], d_range[-1])):
            for q in list(range(q_range[0], q_range[-1])):
                arima_model(filename, date_parser, p, d, q)
    return


def export_mse_list():
    """Exports the mse rating list to a .csv file"""
    mse_file = csv_writer(os.path.join(OUTPUT_FOLDER, "mse_rating.csv"), ["MSE", "Model"])
    mse_file.write_at_once(mse_list)
    mse_file.close()


def export_log_file():
    """Exports the log list to a .txt file"""
    log_file = open(os.path.join(OUTPUT_FOLDER, "log.txt"), "w")
    for line in log_list:
        log_file.write(line)
        log_file.write("\n")
    log_file.close()


def init():
    """Main function"""

    def parser(x):
        return datetime.strptime(f"190{x}", "%Y-%m")

    arima_automated(filename="shampoo-sales.csv", date_parser=parser, p_range=[1, 6], d_range=[0, 4], q_range=[0, 4])
    export_mse_list()
    export_log_file()


init()
