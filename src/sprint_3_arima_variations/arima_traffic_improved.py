import os

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from pandas import DataFrame, concat, read_csv
from statsmodels.tsa.arima_model import ARIMA
from matplotlib import pyplot
from numpy import concatenate
from math import sqrt

# from datetime import datetime

PATH = os.path.join(".", "sprint-3_arima_variations")
os.chdir(PATH)

OUTPUT_FOLDER = "results_arima_2"


# def parser(x):
#     return datetime.strptime(x, "%Y-%m-%d %H:%M:%S+00:00")


def run_tests(train, test, scaler, arima_parameters, num_predictions, location, data_split):
    # Output filename
    outputfilename = f"{location}_{num_predictions}"
    # Save each result
    results_list = list()

    # prediction_list = [-96, -72, -48, -36, -24, -12 ]
    prediction_list = [-72]

    for s_seq in prediction_list:
        history = concatenate((train, test[0:s_seq]), axis=0)
        if s_seq + num_predictions == 0:
            real_results = test[s_seq:]
        else:
            real_results = test[s_seq: s_seq + num_predictions]

        ## Create Tests for next
        for arima_parameter in arima_parameters:
            # print("Results for ARIMA with window %s" % arima_parameter[0])
            print(f"Results for ARIMA with window {arima_parameter[0]}")
            try:
                broke = 0
                model = ARIMA(history, order=(arima_parameter[0], arima_parameter[1], arima_parameter[2]))
                model_fit = model.fit(disp=0)
                output, output_stderr, output_conf_interval = model_fit.forecast(steps=num_predictions)
                output_unscaled = scaler.inverse_transform([output])[0]
                test_unscaled = scaler.inverse_transform([real_results])[0]

            except Exception as e:
                print(str(e))
                broke = 1
                print("Arima Broke!")
                mae_error = 0
                mse_error = 0
                rmse_error = 0

            if broke == 0:
                # raw_test_values = dt1[len(train_scaled)+1:len(train_scaled)+1+len(test)]
                ## Calculate MSE
                mae_error = mean_absolute_error(test_unscaled, output_unscaled)
                mse_error = mean_squared_error(test_unscaled, output_unscaled)
                rmse_error = sqrt(mse_error)
                print("Test MAE: %.3f" % mae_error)
                print("Test MSE: %.3f" % mse_error)
                print("Test RMSE: %.3f" % rmse_error)
                ## Plot Results
                pyplot.figure(figsize=(4, 4))
                pyplot.yscale("linear")
                # pyplot.plot(test_unscaled, color='black')
                # pyplot.plot(output_unscaled, color='red')
                pyplot.plot(range(len(test_unscaled)), test_unscaled, marker="H", color="black", label="Real Values")
                pyplot.plot(range(len(output_unscaled)), output_unscaled, marker="s", color="red",
                            label="Blind Prediction")
                pyplot.ylabel("Speed Difference")
                pyplot.xlabel("Timesteps")
                pyplot.grid(which="major", alpha=0.3, color="#666666", linestyle="-")
                pyplot.ylim([0, 45])
                pyplot.xlim([0, 12])

                # output_file = f"""{outputfilename}_arima({arima_parameter[0]},{arima_parameter[1]},{arima_parameter[2]})
                #             _predictions_{num_predictions}_crossvalidation_{data_split}_test_{s_seq}.png"""
                figure_name = os.path.join(
                    OUTPUT_FOLDER,
                    f"{outputfilename}_arima({arima_parameter[0]},{arima_parameter[1]},{arima_parameter[2]})"
                    f"_predictions_{num_predictions}_crossvalidation_{data_split}_test_{s_seq}.png",
                )

                ## Save Results
                try:
                    pyplot.savefig(figure_name)
                except FileNotFoundError:
                    os.mkdir(OUTPUT_FOLDER)
                    pyplot.savefig(figure_name)

                ## Show
                pyplot.show()
                raw_results = {"predicted": output_unscaled, "real": test_unscaled}
                print(raw_results)
                results_dataset_raw = DataFrame(raw_results)
                results_dataset_raw.to_csv(
                    os.path.join(
                        OUTPUT_FOLDER,
                        f"{outputfilename}_raw_arima({arima_parameter[0]},{arima_parameter[1]},{arima_parameter[2]})"
                        f"_predictions_{num_predictions}_crossvalidation_{data_split}_test_{s_seq}.csv",
                    )
                )

            results_list.append(
                [
                    location,
                    num_predictions,
                    arima_parameter[0],
                    arima_parameter[1],
                    arima_parameter[2],
                    mae_error,
                    mse_error,
                    rmse_error,
                    data_split,
                    s_seq,
                    broke,
                ]
            )

    columns = ["location", "predictedValues", "a1", "a2", "a3", "mae", "mse", "rmse", "split", "test", "broke"]
    results_dataset = DataFrame(results_list, columns=columns)
    return results_dataset


####################################################################
# Experiments
####################################################################
# Load Dataset
## Configure Experiments

# data_offsets = [0,200]
locations = ["Braga"]
predictions = [12]
# arima_windows = [(8,1,1),(8,1,2),(12,1,1),(12,1,2)]
arima_windows = [(12, 1, 1)]

columns = ["location", "predictedValues", "a1", "a2", "a3", "mae", "mse", "rmse", "split", "test", "broke"]
results_dataset = DataFrame(columns=columns)

for location in locations:

    dt1 = read_csv("N14Bosch_2019-04.csv", infer_datetime_format=True, parse_dates=["timestep"], index_col=["timestep"])
    dt1 = dt1[["speed_diff"]]
    scaler = MinMaxScaler()
    dt1[["speed_diff"]] = scaler.fit_transform(dt1[["speed_diff"]])

    dt1 = dt1.speed_diff.values

    tscv = TimeSeriesSplit(n_splits=3)

    data_split = 1

    for train_index, test_index in tscv.split(dt1):
        print(f"TRAIN: {train_index}\nTEST: {test_index}")
        train1, test1 = dt1[train_index], dt1[test_index]

        for num_predictions in predictions:
            results = run_tests(train1.copy(), test1.copy(), scaler, arima_windows, num_predictions, location,
                                data_split)

            results_dataset = concat([results_dataset, results], ignore_index=True)

        data_split += 1

try:
    results_dataset.to_csv(os.path.join(OUTPUT_FOLDER, "ARIMA_results_summary.csv"), index=False)
except FileNotFoundError:
    os.mkdir(OUTPUT_FOLDER)
    results_dataset.to_csv(os.path.join(OUTPUT_FOLDER, "ARIMA_results_summary.csv"), index=False)
