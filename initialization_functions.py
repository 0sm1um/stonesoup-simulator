import numpy as np
import datetime

from stonesoup.types.array import StateVector, CovarianceMatrix
from stonesoup.types.state import GaussianState, EnsembleState

from stonesoup.models.transition.linear import (CombinedLinearGaussianTransitionModel,
                                                ConstantTurn, ConstantVelocity)
from stonesoup.models.measurement.nonlinear import CartesianToElevationBearingRange, CartesianToBearingRange
from stonesoup.models.measurement.linear import LinearGaussian


def generate_initial_2d_states(num_vectors):
    #Initialization
    timestamp = datetime.datetime(2021, 4, 2, 12, 28, 57)
    
    x0=StateVector([0, 1, 0, 1]);                  #initial filter state value
    P0=CovarianceMatrix(np.diag(np.array([0,0.05,0,0.05]))**2);#initial covariance matrix 
        
    InitialEnsemble = EnsembleState.generate_ensemble(mean=x0,
                                                      covar=P0,
                                                      num_vectors=num_vectors)
    KFprior = GaussianState(state_vector=x0, covar=P0, timestamp = timestamp)
    
    EnKFprior = EnsembleState(ensemble=InitialEnsemble,
                            timestamp = timestamp)
    EnSRFprior = EnsembleState(ensemble=InitialEnsemble,
                            timestamp = timestamp)

    import pickle
    with open('initial_2d_state_list.txt', 'wb') as f:
        pickle.dump([KFprior, EnKFprior, EnSRFprior], f)
    return [KFprior, EnKFprior, EnSRFprior]

def generate_initial_states_Niu():
    #Initialization
    timestamp = datetime.datetime(2021, 4, 2, 12, 28, 57)
  
    x0=StateVector([1000, 0, 2650, 150, 200, 0]);                  #initial filter state value
    P0=CovarianceMatrix(np.diag(np.array([100,5,100,5,100,5]))**2);#initial covariance matrix 
       
    num_vectors = 500
    
    InitialEnsemble = EnsembleState.generate_ensemble(mean=x0,
                                                      covar=P0,
                                                      num_vectors=num_vectors)
    EnKFprior = EnsembleState(ensemble=InitialEnsemble,
                            timestamp = timestamp)
    EnSRFprior = EnsembleState(ensemble=InitialEnsemble,
                            timestamp = timestamp)

    return [EnKFprior, EnSRFprior]



def generate_2d_linear_models():
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

    return transition_model, measurement_model

def generate_2d_nonlinear_models():
    q_x = 0.05
    q_y = 0.05
    transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(q_x),
                                                          ConstantVelocity(q_y)])
    sensor_x = 50  # Placing the sensor off-centre
    sensor_y = 0
    
    measurement_model = CartesianToBearingRange(
    ndim_state=4,
    mapping=(0, 2),
    noise_covar=np.diag([np.radians(0.2), 1]),  # Covariance matrix. 0.2 degree variance in
    # bearing and 1 metre in range
    translation_offset=np.array([[sensor_x], [sensor_y]])  # Offset measurements to location of
    # sensor in cartesian.
)

    return transition_model, measurement_model

def generate_models_Niu():
    omg=6/180*np.pi;                                               # Turn Rate For transition model
    sigma_r=28; sigma_theta=0.1; sigma_fa=0.1;                     # standard deviation for measurement noise 
    
    measurement_model = CartesianToElevationBearingRange(ndim_state=6,
        mapping=(0,1,2),
        noise_covar=np.diag(np.array([sigma_theta,sigma_fa,sigma_r])**2))
    
    transition_model = CombinedLinearGaussianTransitionModel([ConstantTurn(
                                                             turn_noise_diff_coeffs=np.zeros([2]),
                                                             turn_rate=omg),
                                                             ConstantVelocity(5)])
    return transition_model, measurement_model


def generate_linear_models_Niu():
    omg=6/180*np.pi;                                               # Turn Rate For transition model
    
    measurement_model = LinearGaussian(ndim_state = 6, mapping = np.array([0,2,4]), noise_covar = 5*np.eye(3))
    
    transition_model = CombinedLinearGaussianTransitionModel([ConstantTurn(
                                                             turn_noise_diff_coeffs=np.diag([5,5]),
                                                             turn_rate=omg),
                                                             ConstantVelocity(noise_diff_coeff=5)])
    return transition_model, measurement_model