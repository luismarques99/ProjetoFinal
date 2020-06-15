from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit

from statsmodels.tsa.statespace.sarimax import SARIMAX

from matplotlib import pyplot
from numpy import array
from numpy import concatenate
from math import sqrt

from pandas.plotting import autocorrelation_plot
from pathlib import Path

import numpy as np
import pandas as pd
import copy



# Date Parser
def parser(x):
    return datetime.strptime(x, '%Y-%m-%d %H:%M:%S+00:00')



def run_tests(train, test, scaler, sarima_order_parameters, sarima_season_parameters, num_predicitons, location, data_split):   
    # Output filename
    outputfilename = location  + "_" + str(num_predictions)
    # Save each result
    results_list = []
    
    prediction_list = [-96, -72, -48, -36, -24, -12 ]

    for s_seq in prediction_list:
        history = np.concatenate((train, test[0:s_seq]),axis=0)
        if (s_seq+num_predicitons ==0):
            real_results = test[s_seq:]
        else:
            real_results = test[s_seq:s_seq+num_predicitons]


        ## Create Tests for next
        for sarima_order_parameter in sarima_order_parameters:
            for sarima_season_parameter in sarima_season_parameters:
                print("Results for SARIMA with window %s" % sarima_order_parameter[0])
                try:
                    broke=0
                    model = SARIMAX(endog=history, order=(sarima_order_parameter[0],sarima_order_parameter[1],sarima_order_parameter[2]),seasonal_order =(sarima_season_parameter[0],sarima_season_parameter[1],sarima_season_parameter[2],sarima_season_parameter[3]),enforce_stationarity=False, enforce_invertibility=False)
                    model_fit = model.fit(disp=0)
                    output = model_fit.forecast(steps=num_predictions)
                    output_unscaled = scaler.inverse_transform([output])[0]
                    test_unscaled = scaler.inverse_transform([real_results])[0]
                    
                except Exception as e:
                    print(str(e))
                    broke = 1
                    print("SArima Broke!")
                    mae_error = 0
                    mse_error = 0
                    rmse_error = 0

                if(broke==0):
                    
                    # raw_test_values = dt1[len(train_scaled)+1:len(train_scaled)+1+len(test)]
                    ## Calculate MSE
                    mae_error = mean_absolute_error(test_unscaled, output_unscaled)
                    mse_error = mean_squared_error(test_unscaled , output_unscaled)
                    rmse_error = sqrt(mse_error)
                    print('Test MAE: %.3f' % mae_error)
                    print('Test MSE: %.3f' % mse_error)
                    print('Test RMSE: %.3f' % rmse_error)
                    ## Plot Results
                    pyplot.figure(figsize=(8,6))
                    pyplot.yscale('linear')
                    pyplot.plot(test_unscaled, color='black')
                    pyplot.plot(output_unscaled, color='red')
                    
                    ## Save Results
                    figure_name = 'results_sarima/'+outputfilename+'_sarima(' + str(sarima_order_parameter[0])+ ',' + str(sarima_order_parameter[1]) + ',' + str(sarima_order_parameter[2]) + ')('+str(sarima_season_parameter[0]) + ','+str(sarima_season_parameter[1]) + ','+str(sarima_season_parameter[2]) + ','+str(sarima_season_parameter[3]) + ','+ ')_predictions_' + str(num_predictions)+'_crossvalidation_'+ str(data_split)+'_test_' + str(s_seq) +'.png' 
                    pyplot.savefig(figure_name)
                    ## Show
                    pyplot.show()
                    raw_results = {'predicted':output_unscaled,'real':test_unscaled}
                    print(raw_results)
                    results_dataset_raw = pd.DataFrame(raw_results)
                    results_dataset_raw.to_csv('results_sarima/'+outputfilename+'_raw_'+'_sarima(' + str(sarima_order_parameter[0])+ ',' + str(sarima_order_parameter[1]) + ',' + str(sarima_order_parameter[2]) + ')('+str(sarima_season_parameter[0]) + ','+str(sarima_season_parameter[1]) + ','+str(sarima_season_parameter[2]) + ','+str(sarima_season_parameter[3]) + ','+')_predictions_' + str(num_predictions)+'_crossvalidation_'+ str(data_split)+'_test_' + str(s_seq) +'.csv')

                results_list.append([location, num_predictions, sarima_order_parameter[0], sarima_order_parameter[1],sarima_order_parameter[2],sarima_season_parameter[0],sarima_season_parameter[1],sarima_season_parameter[2],sarima_season_parameter[3], mae_error, mse_error, rmse_error, data_split ,s_seq ,broke ])
        
    columns = ['location','predictedValues', 'so1', 'so2', 'so3','ss1','ss2','ss3', 'mae', 'mse', 'rmse', 'split','test','broke']
    results_dataset = pd.DataFrame(results_list,columns=columns)
    return results_dataset
    
####################################################################    
#Experiments    
####################################################################    
# Load Dataset
## Configure Experiments

# data_offsets = [0,200]
locations = ["Braga"]
predictions = [12]

sarima_order_parameters = [(8,1,1),(8,1,2),(12,1,1),(12,1,2)]
sarima_season_parameters = [(0,0,1,24),(0,1,1,24),(1,1,1,24),(0,0,1,168),(0,1,1,168),(1,1,1,168)]

columns = ['location','predictedValues', 'so1', 'so2', 'so3','ss1','ss2','ss3', 'mae', 'mse', 'rmse', 'split','test','broke']
results_dataset = pd.DataFrame(columns=columns)

for location in locations:

    dt1 = pd.read_csv(r'dataset/roads/1h/N14Bosch_1h_20190522134810.csv', infer_datetime_format=True, parse_dates=['timestep'], index_col=['timestep'])
    dt1 = dt1[['speed_diff']]
    scaler = MinMaxScaler()
    dt1[['speed_diff']] = scaler.fit_transform(dt1[['speed_diff']])

    dt1 = dt1.speed_diff.values

    tscv = TimeSeriesSplit(n_splits=3)
    
    data_split = 1

    for train_index, test_index in tscv.split(dt1):
        print("TRAIN:", train_index, "TEST:", test_index)
        train1, test1 = dt1[train_index], dt1[test_index]
        
        for num_predictions in predictions:
        
            results = run_tests(train1.copy(),test1.copy(), scaler, sarima_order_parameters, sarima_season_parameters, num_predictions, location, data_split)

            results_dataset=pd.concat([results_dataset,results], ignore_index=True)

        data_split += 1
                
results_dataset.to_csv('results_sarima/SARIMA_results_summary.csv', index=False, )




