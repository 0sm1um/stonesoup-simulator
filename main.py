import numpy as np
import matplotlib.pyplot as plt
import pickle
import datetime

from track_functions import monte_carlo_runs, generate_ground_truth
from initialization_functions import generate_models

from RMSE_functions import calc_RMSE, plot_RMSE

#Initialization
timestamp = datetime.datetime(2021, 4, 2, 12, 28, 57)
tMax =60
time_span = np.array([timestamp + datetime.timedelta(seconds=i) for i in range(tMax)])


try:
    with open('rmse.py', 'rb') as f:
        RMSE_LIST = pickle.load(f)
except:
    print('No RMSE data found, running monte carlo simulations now:')
    monte_carlo_iterations = 25
    
    EnKF_runs,EnSRF_runs,PCEnKF_runs,PCEnSRF_runs,ground_truth=monte_carlo_runs(time_span,
                                                                                monte_carlo_iterations)
    
    RMSE_EnKF = calc_RMSE(ground_truth,EnKF_runs)
    RMSE_EnSRF = calc_RMSE(ground_truth,EnSRF_runs)
    RMSE_PCEnKF = calc_RMSE(ground_truth,PCEnKF_runs)
    RMSE_PCEnSRF = calc_RMSE(ground_truth,PCEnSRF_runs)
    
    RMSE_LIST = [RMSE_EnKF, RMSE_EnSRF, RMSE_PCEnKF, RMSE_PCEnSRF]
    
NAME_LIST = ['Particle Filter', 'EnKF', 'EnSRF', 'PCEnKF', 'PCEnSRF']



#plot_list = plot_RMSE(RMSE_LIST, NAME_LIST)
tm, mm = generate_models()
ground_truth = generate_ground_truth(tm,time_span)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.plot([e.state_vector[0] for e in ground_truth],
        [e.state_vector[2] for e in ground_truth],
        [e.state_vector[4] for e in ground_truth])
