import numpy as np
import matplotlib.pyplot as plt
import pickle
import datetime

from stonesoup.types.array import StateVector, CovarianceMatrix
from stonesoup.types.state import State, GaussianState, EnsembleState

from stonesoup.models.transition.linear import (CombinedLinearGaussianTransitionModel,
                                                ConstantTurn, ConstantVelocity)
from stonesoup.models.measurement.nonlinear import CartesianToElevationBearingRange, CartesianToBearingRange
from stonesoup.models.measurement.linear import LinearGaussian



from track_functions import generate_ground_truth, \
                            linear_2d_monte_carlo_runs, nonlinear_2d_monte_carlo_runs,monte_carlo_runs_Niu
from initialization_functions import generate_models_Niu

from RMSE_functions import calc_RMSE, plot_RMSE
from MISC_functions import pack_RMSE_data

from simulator import simulator

from stonesoup.updater.ensemble import (EnsembleUpdater, EnsembleSqrtUpdater)
from stonesoup.updater.kalman import KalmanUpdater, ExtendedKalmanUpdater
from stonesoup.predictor.ensemble import EnsemblePredictor

'''
    This script represents the code used to gather the data used in [PAPER HERE].
    
    This repository is structured such that different stone soup can be run 
    relativley rapidly. 
    
    The simulator class requires a transition and 
    measurement model, then the simulate_track method accepts a Stone Soup
    Predictor, Updater, ground truth initial state, initial state for the
    chosen algorithm, and a span of time which the simulation takes place over.
    This time span should be an evenly spaced datetime.datetime list.
    
    The simulator then, is used to gather "Track" instances, and with a list 
    of tracks, RMSE can then be calculated.
'''

i = 10
num_vectors = i*5
print('Ensemble_Size for M =',num_vectors,' ensemble size is starting:')
#Initialization
timestamp = datetime.datetime(2021, 4, 2, 12, 28, 57)
tMax = 60
dt = 1
tRange = tMax // dt
plot_time_span = np.array([dt*i for i in range(tRange)])

time_span = np.array([timestamp + datetime.timedelta(seconds=dt*i) for i in range(tRange)])

monte_carlo_iterations = 50


q_x = 0.05
q_y = 0.05
transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(q_x),
                                                      ConstantVelocity(q_y)])
measurement_model = LinearGaussian(
ndim_state=4,  # Number of state dimensions (position and velocity in 2D)
mapping=(0, 2),  # Mapping measurement vector index to state index
noise_covar=np.array([[5, 0],  # Covariance matrix for Gaussian PDF
                      [0, 5]])
)

simulator = simulator(transition_model=transition_model, measurement_model=measurement_model)


predictor = EnsemblePredictor(transition_model)
updater = EnsembleUpdater(measurement_model)

initial_ground_truth = State(state_vector=StateVector([0, 1, 0, 1]),
                             timestamp = time_span[0])

EnKFprior = EnsembleState.from_gaussian_state(
    GaussianState(state_vector=StateVector([0, 1, 0, 1]),
                  covar = CovarianceMatrix(np.diag(np.array([0,0.05,0,0.05]))**2),
                  timestamp = time_span[0]), num_vectors)

EnKF_run = simulator.simulate_track(predictor = predictor, 
                                    updater = updater, 
                                    initial_state = initial_ground_truth,
                                    prior = EnKFprior,
                                    time_span=time_span)

'''
EKF_runs, EnKF_runs, EnSRF_runs, ground_truth = nonlinear_2d_monte_carlo_runs(time_span, monte_carlo_iterations, num_vectors)

RMSE_EKF = calc_RMSE(ground_truth, EKF_runs)
RMSE_EnKF = calc_RMSE(ground_truth,EnKF_runs)
RMSE_EnSRF = calc_RMSE(ground_truth,EnSRF_runs)

RMSE_LIST = [RMSE_EKF, RMSE_EnKF, RMSE_EnSRF]

prefix = 'rmse_size'
suffix = 'Ensemble.txt'
filename = prefix+str(num_vectors)+suffix
pack_RMSE_data(filename, RMSE_LIST)
j=0
for RMSE in RMSE_LIST:
    plot_list = plot_RMSE(RMSE, plot_time_span)
    j = j+1



pack_RMSE_data('track_nonlinear_measurement_Ensemble.txt',

with open('rmse_nonlinear_measurement_Ensemble.txt', 'wb') as f:
    pickle.dump([RMSE_LIST], f)

with open('track_nonlinear_measurement_Ensemble.txt', 'wb') as g:
    pickle.dump([EnKF_runs, EnSRF_runs], g)

'''

