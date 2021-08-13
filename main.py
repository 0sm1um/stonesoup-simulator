"""
@author: John Hiles
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle
import datetime

from stonesoup.types.array import StateVector, CovarianceMatrix
from stonesoup.types.state import State, GaussianState, EnsembleState

from stonesoup.models.transition.linear import (CombinedLinearGaussianTransitionModel,
                                                ConstantVelocity)
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.updater.ensemble import (EnsembleUpdater, EnsembleSqrtUpdater)
from stonesoup.updater.kalman import KalmanUpdater, ExtendedKalmanUpdater
from stonesoup.predictor.ensemble import EnsemblePredictor
from stonesoup.predictor.kalman import ExtendedKalmanPredictor


from RMSE_functions import calc_RMSE, plot_RMSE
from MISC_functions import pack_RMSE_data

from simulator import simulator


"""
    This script represents the code used to gather the data used in [PAPER HERE].
    
    This repository is structured such that different stone soup algorithms 
    can be run rapidly. Hopefully I've made it modular enough to 
    allow swapping of things like algorithms, and "experiments" by replacing
    the desired transition and measurement models.
    
    The simulator class requires a transition and 
    measurement model, then the simulate_track method accepts a Stone Soup
    Predictor, Updater, ground truth initial state, initial state for the
    chosen algorithm, and a span of time which the simulation takes place over.
    This time span should be an evenly spaced datetime.datetime list.
    
    The simulator then, is used to gather "Track" instances, and with a list 
    of tracks, RMSE can then be calculated.
"""

i = 10
num_vectors = i*5

"""
    Here, we get our initial variables for simulation. For this, we are just
    using a time span of 60 time instances spaced one second apart.
"""

timestamp = datetime.datetime(2021, 4, 2, 12, 28, 57)
tMax = 60
dt = 1
tRange = tMax // dt
plot_time_span = np.array([dt*i for i in range(tRange)])

time_span = np.array([timestamp + datetime.timedelta(seconds=dt*i) for i in range(tRange)])

monte_carlo_iterations = 5




"""
Here we instantiate our transition and measurement models. These are the 
same models used in the StoneSoup Kalman Filter examples.
"""

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

"""
Here we instantiate the simulator with our transition and measurement model.
This class is capable of generating sets of ground truth points, and simulate
measurements for our recursive filters.
"""

simulator = simulator(transition_model=transition_model,
                      measurement_model=measurement_model)

"""
Finally, before running our monte carlo simulation, we need to instantiate
all the algorithm predictor/updaters we wish to run
"""

EKFpredictor = ExtendedKalmanPredictor(transition_model)
EKFupdater = ExtendedKalmanUpdater(measurement_model)
EnKFpredictor = EnsemblePredictor(transition_model)
EnKFupdater = EnsembleUpdater(measurement_model)
EnSRFpredictor = EnsemblePredictor(transition_model)
EnSRFupdater = EnsembleSqrtUpdater(measurement_model)

"""
Now, we will provide initial states for the ground truth, and the value we 
initialize our algorithms with. It is of course optional to initialize the 
algorithms with the same value as the ground truth, and I would actually 
encourage you not to do this.
"""

initial_ground_truth = State(state_vector=StateVector([0, 1, 0, 1]),
                             timestamp = time_span[0])
EKFprior = GaussianState(state_vector=StateVector([0, 1, 0, 1]),
              covar = CovarianceMatrix(np.diag(np.array([0,0.05,0,0.05]))**2),
              timestamp = time_span[0])
EnKFprior = EnsembleState.from_gaussian_state(EKFprior, num_vectors)
EnSRFprior = EnsembleState.from_gaussian_state(EKFprior, num_vectors)

"""
Finally, we run our Simulations many times. By populating a list with tracks
of simulation instances, we can compute Root Mean Squared by averaging the 
difference between the ground truth and our algorithm's predictions across 
many simulation runs, hence it is a Monte Carlo approach.
"""

EKF_monte_carlo_runs = []
EnKF_monte_carlo_runs = []
EnSRF_monte_carlo_runs = []
for i in range(monte_carlo_iterations):
    print(i+1)
    EKF_monte_carlo_runs.append(simulator.simulate_track(predictor = EKFpredictor, 
                                        updater = EKFupdater, 
                                        initial_state = initial_ground_truth,
                                        prior = EKFprior,
                                        time_span=time_span))
    EnKF_monte_carlo_runs.append(simulator.simulate_track(predictor = EnKFpredictor, 
                                        updater = EnKFupdater, 
                                        initial_state = initial_ground_truth,
                                        prior = EnKFprior,
                                        time_span=time_span))
    EnSRF_monte_carlo_runs.append(simulator.simulate_track(predictor = EnSRFpredictor, 
                                        updater = EnSRFupdater, 
                                        initial_state = initial_ground_truth,
                                        prior = EnKFprior,
                                        time_span=time_span))

"""
Now we have three lists of M tracks for each algorithm we wish to evaluate.
To compute RMSE, I am just going to call our purpose made function.

Note, its arguments are for the lists of ground truth paths, and for their
associated simulated tracks. For our variables, the 0th and 1st indicies 
correspond to these.
"""

RMSE_EKF = calc_RMSE(EKF_monte_carlo_runs[0],EKF_monte_carlo_runs[1])
RMSE_EnKF = calc_RMSE(EnKF_monte_carlo_runs[0],EnKF_monte_carlo_runs[1])
RMSE_EnSRF = calc_RMSE(EnSRF_monte_carlo_runs[0],EnSRF_monte_carlo_runs[1])

RMSE_LIST = [RMSE_EKF, RMSE_EnKF, RMSE_EnSRF]

"""
Now that we have our RMSE data, it is a matter of plotting it. For the paper,
the authors collaborated remotley, and so the above results were saved by one
author and sent to the other, where they were plotted.

The code below was written to automate the packaging of simulations run with
varying ensemble sizes. Change ensemble size at the start of the script, run
it, and a file is saved with the appropriate name.
"""


for RMSE in RMSE_LIST:
    plot_list = plot_RMSE(RMSE, plot_time_span)
    
prefix = 'rmse_size'
suffix = 'Ensemble.txt'
filename = prefix+str(num_vectors)+suffix
pack_RMSE_data(filename, RMSE_LIST)

"""
This final block of code, serves as a quick preview of the RMSE results for
someone running simulations. This preview window is especially useful for 
debugging filters in development, as you can see at a glance if an algorithm
begins to diverge or perform worse than expected.
"""

plot_time_span = np.array([dt*i for i in range(tRange)])
j=0
for RMSE in RMSE_LIST:
    plot_list = plot_RMSE(RMSE, plot_time_span)
    j = j+1
    
