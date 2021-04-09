import numpy as np
import matplotlib.pyplot as plt
import pickle
import datetime

from track_functions import monte_carlo_runs

from RMSE_functions import calc_RMSE, plot_RMSE

#Initialization
timestamp = datetime.datetime(2021, 4, 2, 12, 28, 57)
tMax =60
time_span = np.array([timestamp + datetime.timedelta(seconds=i) for i in range(tMax)])


monte_carlo_iterations = 2

EnKF_runs,EnSRF_runs,PCEnKF_runs,PCEnSRF_runs,ground_truth=monte_carlo_runs(time_span,
                                                                            monte_carlo_iterations)

RMSE_EnKF = calc_RMSE(ground_truth,EnKF_runs)
RMSE_EnSRF = calc_RMSE(ground_truth,EnSRF_runs)
RMSE_PCEnKF = calc_RMSE(ground_truth,PCEnKF_runs)
RMSE_PCEnSRF = calc_RMSE(ground_truth,PCEnSRF_runs)

RMSE_LIST = [RMSE_EnKF, RMSE_EnSRF, RMSE_PCEnKF, RMSE_PCEnSRF]

NAME_LIST = ['Particle Filter', 'EnKF', 'EnSRF', 'PCEnKF', 'PCEnSRF']


'''
with open('rmse.py', 'wb') as f:
    pickle.dump(RMSE_LIST, f)
    
plot_list = plot_RMSE(RMSE_LIST, NAME_LIST)


import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot([e.state_vector[0] for e in ground_truth[0]],[e.state_vector[2] for e in ground_truth[0]],
'''