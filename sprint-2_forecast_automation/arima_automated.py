import os
import shutil
import csv
from pandas import read_csv
from pandas import DataFrame
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from numpy.linalg import LinAlgError
# from csv_writer import csv_writer

# PATH = os.path.join(".", "sprint-2_forecast_automation")
# os.chdir(PATH)


class arima_model:
    """Class that represents the structure of my automated ARIMA model"""

    OUTPUT_FOLDER = "output"
    TRAIN_SIZE = 0.66

    """Resets the output folder"""
    try:
        shutil.rmtree(OUTPUT_FOLDER)
    except FileNotFoundError:
        pass

    def __init__(self, filename, date_parser, p, d, q):
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
        # self.folder = os.path.join(OUTPUT_FOLDER, f"ARIMA({self.p},{self.d},{self.q})")
        self.dataset = get_series(filename, date_parser)
        self.name = dataset.name
        self.values = dataset.values
        self.train_size = int(len(self.values) * self.TRAIN_SIZE)
        self.train = self.values[0 : self.train_size]
        self.test = self.values[self.train_size : len(self.values)]
        self.history = [x for x in self.train]
        self.predictions = list()
        self.folder = self.create_folder()
        self.file = self.open_file()
        self.execute()

    def execute(self):
        """Executes the model"""

        # self.file.write(f"» ARIMA({self.p}, {self.d} , {self.q}) model predictions «\n")
        try:
            for t in range(len(self.test)):
                model = ARIMA(self.history, order=(self.p, self.d, self.q))
                model_fit = model.fit(disp=0)
                output = model_fit.forecast()
                yhat = output[0]
                self.predictions.append(yhat)
                obs = self.test[t]
                self.history.append(obs)
                self.file.write(f"\npredicted={yhat}, expected={obs}")

            error = mean_squared_error(self.test, self.predictions)
            self.file.write(f"\n\n\nTest MSE: {format(error, '0.3f')}")
            self.export_plot()

        except Exception as err:
            print(f"The model ARIMA({self.p}, {self.d} , {self.q}) raised a {type(err).__name__}: {err}")
            self.file.close()
            shutil.rmtree(self.folder)
            # Imprimir em log file que o modelo não foi aprovado
            pass

        else:
            # Imprimir em log file que o modelo foi aprovado
            pass

        # except ValueError as err:
        #     print(f"The model ARIMA({self.p}, {self.d} , {self.q}) raised a ValueError: {err}")
        #     self.file.close()
        #     shutil.rmtree(self.folder)
        #     pass

        # except LinAlgError as err:
        #     print(f"The model ARIMA({self.p}, {self.d} , {self.q}) raised a LinAlgError: {err}")
        #     self.file.close()
        #     shutil.rmtree(self.folder)
        #     pass

        finally:
            self.file.close()

        print(f"The model ARIMA({self.p}, {self.d} , {self.q}) was successfully exported!")

    def create_folder(self):
        """Creates a folder for the model

        Returns:
            string: folder path
        """
        single_folder = f"ARIMA({self.p},{self.d},{self.q})"
        folder = os.path.join(self.OUTPUT_FOLDER, single_folder)
        try:
            os.mkdir(folder)
        except FileNotFoundError:
            os.mkdir(self.OUTPUT_FOLDER)
            os.mkdir(folder)
        except FileExistsError:
            shutil.rmtree(folder)
            os.mkdir(folder)
        return folder

    def open_file(self):
        """Creates and opens a writing file for the model

        Returns:
            file: file ready to write
        """
        file_name = f"ARIMA({self.p},{self.d},{self.q})-info.txt"
        file_path = os.path.join(self.folder, file_name)
        file = open(file_path, "w")
        return file

    def export_plot(self):
        """Exports the plot to a folder"""
        pyplot.plot(self.train, color="blue")
        pyplot.plot([None for i in self.train] + [x for x in self.test], color="green")
        pyplot.plot([None for i in self.train] + [x for x in self.predictions], color="red")
        pyplot.gcf().canvas.set_window_title(f"ARIMA({self.p}, {self.d}, {self.q})")
        pyplot.savefig(os.path.join(self.folder, f"ARIMA({self.p},{self.d},{self.q})-plot.png"))


# class csv_writer:
#     """Class that represents the structure of a CSV file writer"""

#     def __init__(self, filename="", header=list()):
#         """Creates an instance of a CSV file writer

#         Args:
#             filename (str, required): string to define the name of the file. Defaults to "" (empty string).
#             header (str[], required): list of strings to be written as the header of the file. Defaults to list() (empty list).
#         """
#         self.file = open(filename, "w")
#         self.write_line(header)

#     def write_line(self, content=list()):
#         """Writes a list of strings as a line in the file

#         Args:
#             content (str[], required): list of strings to be written into the file. Defaults to list() (empty list).
#         """
#         self.file.write(content[0])
#         for index in range(len(content) - 1):
#             self.file.write(",")
#             self.file.write(content[index + 1])
#         self.file.write("\n")

#     def close(self):
#         """Closes the file"""
#         self.file.close()


def get_series(filename="", date_parser=None):
    """Set the series

    Args:
        filename (str, opcional): name of the file to read (the file must be inside
                                  the folder 'files'). Defaults to "" (empty string).
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
