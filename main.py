import numpy as np
import matplotlib.pyplot as plt
import pickle
import datetime

from track_functions import generate_ground_truth, \
                            linear_2d_monte_carlo_runs, nonlinear_2d_monte_carlo_runs,monte_carlo_runs_Niu
from initialization_functions import generate_models_Niu

from RMSE_functions import calc_RMSE, plot_RMSE
from MISC_functions import pack_RMSE_data


for i in range(3):
    #Initialization
    timestamp = datetime.datetime(2021, 4, 2, 12, 28, 57)
    tMax = 60
    dt = 1
    tRange = tMax // dt
    plot_time_span = np.array([dt*i for i in range(tRange)])
    
    time_span = np.array([timestamp + datetime.timedelta(seconds=dt*i) for i in range(tRange)])
    
    monte_carlo_iterations = i*5
    
    EnKF_runs, EnSRF_runs, ground_truth = linear_2d_monte_carlo_runs(time_span, monte_carlo_iterations)
    
    RMSE_KF = calc_RMSE(ground_truth,EnKF_runs)
    RMSE_EnKF = calc_RMSE(ground_truth,EnKF_runs)
    RMSE_EnSRF = calc_RMSE(ground_truth,EnSRF_runs)
    
    RMSE_LIST = [RMSE_KF, RMSE_EnKF, RMSE_EnSRF]
    filename = '_Runs_Nonlinear_Measurement_Model.txt'
    filename = i+filename
    pack_RMSE_data(filename, RMSE_LIST)


i=0
for RMSE in RMSE_LIST:
    plot_list = plot_RMSE(RMSE, plot_time_span)
    i = i+1


'''
pack_RMSE_data('track_nonlinear_measurement_Ensemble.txt',

with open('rmse_nonlinear_measurement_Ensemble.txt', 'wb') as f:
    pickle.dump([RMSE_LIST], f)

with open('track_nonlinear_measurement_Ensemble.txt', 'wb') as g:
    pickle.dump([EnKF_runs, EnSRF_runs], g)

'''