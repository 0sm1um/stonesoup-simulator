import numpy as np
import datetime
import time


from stonesoup.models.transition.linear import (CombinedLinearGaussianTransitionModel,
                                                ConstantTurn, ConstantVelocity)
from stonesoup.models.measurement.nonlinear import CartesianToElevationBearingRange

from stonesoup.types.array import StateVector, CovarianceMatrix
from stonesoup.types.state import EnsembleState, PolynomialChaosEnsembleState
from stonesoup.types.polynomialchaosexpansion import PolynomialChaosExpansion


from stonesoup.types.particle import Particle
from stonesoup.types.numeric import Probability  # Similar to a float type
from stonesoup.types.state import ParticleState


from track_functions import generate_ground_truth, PF_Track, EnKF_Track, \
                             EnSRF_Track, PCEnKF_Track, PCEnSRF_Track


#clear all
#clc
#true



 
 #Initialization
timestamp = datetime.datetime(2021, 4, 2, 12, 28, 57)
Tmax =60; dt=1; dt = datetime.timedelta(seconds=1);
time_span = np.array([timestamp + datetime.timedelta(seconds=i) for i in range(Tmax)]);

omg=6/180*np.pi;                                               # Turn Rate For transition model
x0=StateVector([1000, 0, 2650, 150, 200, 0]);                  #initial filter state value
P0=CovarianceMatrix(np.diag(np.array([100,5,100,5,100,5]))**2);#initial covariance matrix 
xt0=x0+np.sqrt(P0)@np.random.normal(1,1,6).reshape(([6,1]));   #ground truth state values 
sigma_r=28; sigma_theta=0.1; sigma_fa=0.1;                     # standard deviation for measurement noise 

measurement_model = CartesianToElevationBearingRange(ndim_state=6,
    mapping=(0,1,2),
    noise_covar=np.diag(np.array([sigma_theta,sigma_fa,sigma_r])**2))

num_vectors = 250

InitialEnsemble = EnsembleState.generate_ensemble(mean=x0,
                                                  covar=P0,
                                                  num_vectors=num_vectors)
EnKFprior = EnsembleState(ensemble=InitialEnsemble,
                        timestamp = time_span[0])
EnSRFprior = EnsembleState(ensemble=InitialEnsemble,
                        timestamp = time_span[0])
PCEnKFprior = PolynomialChaosEnsembleState(ensemble=InitialEnsemble,
                                            timestamp = time_span[0],
                                            expansion = PolynomialChaosExpansion(InitialEnsemble))
PCEnSRFprior = PolynomialChaosEnsembleState(ensemble=InitialEnsemble,
                                            timestamp = time_span[0],
                                            expansion = PolynomialChaosExpansion(InitialEnsemble))



number_particles = num_vectors

# Sample from the prior Gaussian distribution
particles = [Particle(sample.reshape(-1, 1), weight=Probability(1/number_particles))
             for sample in InitialEnsemble.T]

# Create prior particle state.
PFprior = ParticleState(particles, timestamp=time_span[0])

transition_model = CombinedLinearGaussianTransitionModel([ConstantTurn(
                                                         turn_noise_diff_coeffs=np.zeros([2]),
                                                         turn_rate=omg),
                                                         ConstantVelocity(5)])



Monte_Carlo_Iterations = 100

ParticleFilter = []
EnsembleKalmanFilter = []
EnsembleSqrtFilter = []
PCEnsembleFilter = []
PCEnsembleSqrtFilter = []

for i in range(Monte_Carlo_Iterations):
    tic = time.perf_counter()
    ground_truth = generate_ground_truth(transition_model,time_span)
    ParticleFilter.append(PF_Track(ground_truth, transition_model,measurement_model,PFprior))
    EnsembleKalmanFilter.append(EnKF_Track(ground_truth, transition_model,measurement_model,EnKFprior))
    EnsembleSqrtFilter.append(EnSRF_Track(ground_truth, transition_model,measurement_model,EnSRFprior))
    PCEnsembleFilter.append(PCEnKF_Track(ground_truth, transition_model,measurement_model,PCEnKFprior))
    PCEnsembleSqrtFilter.append(PCEnSRF_Track(ground_truth, transition_model,measurement_model,PCEnSRFprior))
    toc = time.perf_counter()
    print(f"One iteration complete in {toc - tic:0.4f} seconds")
    
    
    
    
    
    
    
    
    
    
    
    