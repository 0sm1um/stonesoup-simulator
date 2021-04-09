import numpy as np
import datetime

from stonesoup.types.array import StateVector, CovarianceMatrix
from stonesoup.types.state import EnsembleState, PolynomialChaosEnsembleState
from stonesoup.types.polynomialchaosexpansion import PolynomialChaosExpansion

from stonesoup.models.transition.linear import (CombinedLinearGaussianTransitionModel,
                                                ConstantTurn, ConstantVelocity)
from stonesoup.models.measurement.nonlinear import CartesianToElevationBearingRange


def generate_initial_states():
    #Initialization
    timestamp = datetime.datetime(2021, 4, 2, 12, 28, 57)
    Tmax =60;
    time_span = np.array([timestamp + datetime.timedelta(seconds=i) for i in range(Tmax)]);
    
    x0=StateVector([1000, 0, 2650, 150, 200, 0]);                  #initial filter state value
    P0=CovarianceMatrix(np.diag(np.array([100,5,100,5,100,5]))**2);#initial covariance matrix 
       
    num_vectors = 500
    
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
    
    import pickle
    with open('initial_state_list.py', 'wb') as f:
        pickle.dump([EnKFprior, EnSRFprior, PCEnKFprior, PCEnSRFprior], f)
    return [EnKFprior, EnSRFprior, PCEnKFprior, PCEnSRFprior]

def generate_models():
    omg=6/180*np.pi;                                               # Turn Rate For transition model
    sigma_r=28; sigma_theta=0.1; sigma_fa=0.1;                     # standard deviation for measurement noise 
    
    measurement_model = CartesianToElevationBearingRange(ndim_state=6,
        mapping=(0,1,2),
        noise_covar=np.diag(np.array([sigma_theta,sigma_fa,sigma_r])**2))
    
    transition_model = CombinedLinearGaussianTransitionModel([ConstantTurn(
                                                             turn_noise_diff_coeffs=np.zeros([2]),
                                                             turn_rate=omg),
                                                             ConstantVelocity(50)])
    return transition_model, measurement_model