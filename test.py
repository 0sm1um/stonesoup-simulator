import numpy as np
import datetime
import time

from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.types.detection import Detection
from stonesoup.models.transition.linear import (CombinedLinearGaussianTransitionModel,
                                                ConstantTurn, ConstantVelocity)
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.models.measurement.nonlinear import CartesianToElevationBearingRange

from stonesoup.types.array import StateVector, CovarianceMatrix
from stonesoup.types.state import EnsembleState, PolynomialChaosEnsembleState
from stonesoup.types.polynomialchaosexpansion import PolynomialChaosExpansion

from stonesoup.types.track import Track
from stonesoup.predictor.ensemble import EnsemblePredictor,PolynomialChaosEnsemblePredictor
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.updater.ensemble import (EnsembleUpdater, EnsembleSqrtUpdater,
             PolynomialChaosEnsembleUpdater, PolynomialChaosEnsembleSqrtUpdater)

from stonesoup.types.particle import Particle
from stonesoup.types.numeric import Probability  # Similar to a float type
from stonesoup.types.state import ParticleState
from stonesoup.predictor.particle import ParticlePredictor
from stonesoup.resampler.particle import SystematicResampler
from stonesoup.updater.particle import ParticleUpdater


#clear all
#clc
#true



 
 #Initialization
timestamp = datetime.datetime(2021, 4, 2, 12, 28, 57)
Tmax =60; dt=1; dt = datetime.timedelta(seconds=1);
tt = np.array([timestamp + datetime.timedelta(seconds=i) for i in range(50)]);
Nt=tt.shape[0];   #Time related variables



omg=6/180*np.pi;                                              # Turn Rate
x0=StateVector([1000, 0, 2650, 150, 200, 0]);                  #initial filter state value
P0=CovarianceMatrix(np.diag(np.array([100,5,100,5,100,5]))**2);                 #initial covariance matrix 
xt0=x0+np.sqrt(P0)@np.random.normal(1,1,6).reshape(([6,1]));       #true ground truth state values 
sigma_r=28; sigma_theta=0.1; sigma_fa=0.1;                     # standard deviation for measurement noise 

measurement_model = CartesianToElevationBearingRange(ndim_state=6,
    mapping=(0,1,2),
    noise_covar=np.diag(np.array([sigma_theta,sigma_fa,sigma_r])**2))
'''
measurement_model = LinearGaussian(
    ndim_state=6,  # Number of state dimensions (position and velocity in 2D)
    mapping=(0,1,2,3,4,5),  # Mapping measurement vector index to state index
    noise_covar=np.zeros([6,6])
    )
'''
    

Q = np.zeros([6,6]);                                           # process covariance matrix

num_vectors = 250

InitialEnsemble = PolynomialChaosEnsembleState.generate_ensemble(mean=x0,
                                                                 covar=P0,
                                                                 num_vectors=num_vectors)

EnKFprior = EnsembleState(ensemble=InitialEnsemble,
                        timestamp = tt[0])
EnSRFprior = EnsembleState(ensemble=InitialEnsemble,
                        timestamp = tt[0])

PCEnKFprior = PolynomialChaosEnsembleState(ensemble=InitialEnsemble,
                                            timestamp = tt[0],
                                            expansion = PolynomialChaosExpansion(InitialEnsemble))
PCEnSRFprior = PolynomialChaosEnsembleState(ensemble=InitialEnsemble,
                                            timestamp = tt[0],
                                            expansion = PolynomialChaosExpansion(InitialEnsemble))



number_particles = num_vectors

# Sample from the prior Gaussian distribution
particles = [Particle(sample.reshape(-1, 1), weight=Probability(1/number_particles))
             for sample in InitialEnsemble.T]

# Create prior particle state.
PFprior = ParticleState(particles, timestamp=tt[0])


truth = GroundTruthPath([GroundTruthState(xt0, timestamp=tt[0])])
transition_model = CombinedLinearGaussianTransitionModel([ConstantTurn(
                                                         turn_noise_diff_coeffs=np.zeros([2]),
                                                         turn_rate=omg),
                                                         ConstantVelocity(5)])

for k in range(1, len(tt)):
    truth.append(GroundTruthState(
        transition_model.function(truth[k-1], noise=True, time_interval=datetime.timedelta(seconds=1)),
        timestamp=tt[k-1]))

#Simulate Measurements
measurements = []
for state in truth:
    measurement = measurement_model.function(state, noise=True)
    measurements.append(Detection(measurement,
                                  timestamp=state.timestamp,
                                  measurement_model=measurement_model))
    
EnKFpredictor = EnsemblePredictor(transition_model)
EnKFupdater = EnsembleUpdater(measurement_model)
EnSRFpredictor = EnsemblePredictor(transition_model)
EnSRFupdater = EnsembleUpdater(measurement_model)
PCEnKFpredictor = PolynomialChaosEnsemblePredictor(transition_model)
PCEnKFupdater = PolynomialChaosEnsembleUpdater(measurement_model)
PCEnSRFpredictor = PolynomialChaosEnsemblePredictor(transition_model)
PCEnSRFupdater = PolynomialChaosEnsembleSqrtUpdater(measurement_model)

PFpredictor = ParticlePredictor(transition_model)
resampler = SystematicResampler()
updater = ParticleUpdater(measurement_model, resampler)

num=0
EnKFtrack = Track()
EnSRFtrack = Track()
PCEnKFtrack = Track()
PCEnSRFtrack = Track()
PFtrack = Track()

for measurement in measurements:
    
    tic = time.perf_counter()
    EnKFprediction = EnKFpredictor.predict(EnKFprior, timestamp=measurement.timestamp)
    EnKFhypothesis = SingleHypothesis(EnKFprediction, measurement)  # Group a prediction and measurement
    EnKFposterior = EnKFupdater.update(EnKFhypothesis)
    EnKFtrack.append(EnKFposterior)
    EnKFprior = EnKFtrack[-1]
    toc = time.perf_counter()
    print(f"EnKF in {toc - tic:0.4f} seconds")
    
    tic = time.perf_counter()
    EnSRFprediction = EnKFpredictor.predict(EnSRFprior, timestamp=measurement.timestamp)
    EnSRFhypothesis = SingleHypothesis(EnSRFprediction, measurement)  # Group a prediction and measurement
    EnSRFposterior = EnSRFupdater.update(EnSRFhypothesis)   
    EnSRFtrack.append(EnSRFposterior)
    EnSRFprior = EnSRFtrack[-1]
    toc = time.perf_counter()
    print(f"EnSRF in {toc - tic:0.4f} seconds")
    
    
    tic = time.perf_counter()
    PCEnKFprediction = PCEnKFpredictor.predict(PCEnKFprior, timestamp=measurement.timestamp)
    PCEnKFhypothesis = SingleHypothesis(PCEnKFprediction, measurement)  # Group a prediction and measurement
    PCEnKFposterior = PCEnKFupdater.update(PCEnKFhypothesis)    
    PCEnKFtrack.append(PCEnKFposterior)  
    PCEEnKFprior = PCEnKFtrack[-1]
    toc = time.perf_counter()
    print(f"PCEnKF in {toc - tic:0.4f} seconds")
    
    
    tic = time.perf_counter()
    PCEnSRFprediction = PCEnKFpredictor.predict(PCEnSRFprior, timestamp=measurement.timestamp)
    PCEnSRFhypothesis = SingleHypothesis(PCEnSRFprediction, measurement)  # Group a prediction and measurement
    PCEnSRFposterior = PCEnSRFupdater.update(PCEnSRFhypothesis)
    PCEnSRFtrack.append(PCEnSRFposterior)    
    PCEEnSRFprior = PCEnSRFtrack[-1]
    toc = time.perf_counter()
    print(f"PCEnSRF in {toc - tic:0.4f} seconds")
    
    tic = time.perf_counter()
    PFprediction = PFpredictor.predict(PFprior, timestamp=measurement.timestamp)
    PFhypothesis = SingleHypothesis(PFprediction, measurement)
    PFposterior = updater.update(PFhypothesis)
    PFtrack.append(PFposterior)
    PFprior = PFtrack[-1]
    num = num+1
    toc = time.perf_counter()
    print(f"Particle Filter in {toc - tic:0.4f} seconds")
    
    
    print('-----------------')
    print(num)
    print('-----------------')

"""
#CKF (cubature kalman filter) 
xe_ckf=zeros(6,Nt);  xe_ckf(:,1)=x0; # initial ckf_state values 
P_ckf=P0; # initila ckf_covariance matrix 

#EKF (extended kalman filter)
xe_ekf=xe_ckf; % initial ekf_state values 
P_ekf=P0; % initila ekf_covariance matrix 
'''

'''
# SREnkf (square root ensmble kalman filter)
generate; #generate ensembles points
# Generate_points_on_basis_ensemble(points_ensemble_SREnkf,0);
xe_SREnkf=xe_ckf;  xe_SREnkf(:,1)=x0; # initial SREnkf_state values 
x_points_SREnkf=sqrt(P0)*points_SREnkf+x0; # generate ensemble of sample points (a random variable following the multi-dimensional standard
                                           # Gaussian distribution)

# generate prediction step, measurement model and update sate for each different filtering
# such as (ekf, ckf, srenkf) 
for t=1:(Nt-1)
       xt(:,t+1) = dyn(xt(:,t),dt) + Q*randn(6,1); # dynamic model (prediction step)
       z         = meas(xt(:,t+1))+sqrtm(R)*randn(3,1); # measurement model 
       
# update state using ekf        
# [xe_ekf(:,t+1),P_ekf] = Kalman_ekf(dt,xe_ekf(:,t),P_ekf,z,R);
 
# update state using ckf     
  [xe_ckf(:,t+1),P_ckf] = Kalman_ckf(dt,xe_ckf(:,t),P_ckf,z,R); 

# update state using SREnkf    
# [xe_Enkf(:,t+1),x_points_Enkf] = gPC_SREnkf2(dt,x_points_Enkf,z,R);
  [xe_SREnkf(:,t+1),x_points_SREnkf] = gPC_SREnkf(dt,x_points_SREnkf,z,R);
      
end

# figure 
# plot3(xt(1,:),xt(3,:),xt(5,:))
# grid
"""