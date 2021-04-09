import numpy as np
import matplotlib.pyplot as plt
import time
import pickle

from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.types.detection import Detection
from stonesoup.types.array import StateVector, StateVectors, CovarianceMatrix

from stonesoup.types.track import Track
from stonesoup.predictor.ensemble import EnsemblePredictor,PolynomialChaosEnsemblePredictor
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.updater.ensemble import (EnsembleUpdater, EnsembleSqrtUpdater,
             PolynomialChaosEnsembleUpdater, PolynomialChaosEnsembleSqrtUpdater)

from stonesoup.predictor.particle import ParticlePredictor
from stonesoup.resampler.particle import SystematicResampler
from stonesoup.updater.particle import ParticleUpdater

from initialization_functions import generate_initial_states, generate_models

def generate_ground_truth(transition_model,time_span):
    time_interval = time_span[1]-time_span[0]
    x0=StateVector([1000, 0, 2650, 150, 200, 0])
    P0=CovarianceMatrix(np.diag(np.array([100,5,100,5,100,5]))**2);#initial covariance matrix 
    xt0=x0+np.sqrt(P0)@np.random.normal(1,1,6).reshape(([6,1]));
    truth = GroundTruthPath([GroundTruthState(xt0,
                                              timestamp=time_span[0])])
    for k in range(1, len(time_span)):
        truth.append(GroundTruthState(
            transition_model.function(truth[k-1], noise=True, time_interval=time_interval),
            timestamp=time_span[k-1]))
    return truth

def simulate_measurements(ground_truth,measurement_model):
    #Simulate Measurements
    measurements = []
    for state in ground_truth:
        measurement = measurement_model.function(state, noise=True)
        measurements.append(Detection(measurement,
                                      timestamp=state.timestamp,
                                      measurement_model=measurement_model))
    return measurements

def PF_Track(ground_truth, transition_model, measurement_model, prior):    
    #Simulate Measurements
    measurements = simulate_measurements(ground_truth,measurement_model)
    
    #Create Predictor and Updater
    predictor = ParticlePredictor(transition_model)
    resampler = SystematicResampler()
    updater = ParticleUpdater(measurement_model,resampler)
    
    #Initialize Loop Variables
    track = Track()
    for measurement in measurements:
        prediction = predictor.predict(prior, timestamp=measurement.timestamp)
        hypothesis = SingleHypothesis(prediction, measurement)  # Group a prediction and measurement
        posterior = updater.update(hypothesis)
        track.append(posterior)
        prior = track[-1]

    return track

def EnKF_Track(ground_truth, transition_model, measurement_model, prior):
    #Simulate Measurements
    measurements = simulate_measurements(ground_truth,measurement_model)
    
    #Create Predictor and Updater
    predictor = EnsemblePredictor(transition_model)
    updater = EnsembleUpdater(measurement_model)
    
    #Initialize Loop Variables
    track = Track()
    for measurement in measurements:
        prediction = predictor.predict(prior, timestamp=measurement.timestamp)
        hypothesis = SingleHypothesis(prediction, measurement)  # Group a prediction and measurement
        posterior = updater.update(hypothesis)
        track.append(posterior)
        prior = track[-1]

    return track

def EnSRF_Track(ground_truth, transition_model, measurement_model, prior):
    #Simulate Measurements
    measurements = simulate_measurements(ground_truth,measurement_model)
    
    #Create Predictor and Updater
    predictor = EnsemblePredictor(transition_model)
    updater = EnsembleSqrtUpdater(measurement_model)
    
    #Initialize Loop Variables
    track = Track()
    for measurement in measurements:
        prediction = predictor.predict(prior, timestamp=measurement.timestamp)
        hypothesis = SingleHypothesis(prediction, measurement)  # Group a prediction and measurement
        posterior = updater.update(hypothesis)
        track.append(posterior)
        prior = track[-1]
        
    return track

def PCEnKF_Track(ground_truth, transition_model, measurement_model, prior):
    #Simulate Measurements
    measurements = simulate_measurements(ground_truth,measurement_model)
    
    #Create Predictor and Updater
    predictor = PolynomialChaosEnsemblePredictor(transition_model)
    updater = PolynomialChaosEnsembleUpdater(measurement_model)
    
    #Initialize Loop Variables
    track = Track()
    for measurement in measurements:
        prediction = predictor.predict(prior, timestamp=measurement.timestamp)
        hypothesis = SingleHypothesis(prediction, measurement)  # Group a prediction and measurement
        posterior = updater.update(hypothesis)
        track.append(posterior)
        prior = track[-1]
        
    return track

def PCEnSRF_Track(ground_truth, transition_model, measurement_model, prior):
    #Simulate Measurements
    measurements = simulate_measurements(ground_truth,measurement_model)
    
    #Create Predictor and Updater
    predictor = PolynomialChaosEnsemblePredictor(transition_model)
    updater = PolynomialChaosEnsembleSqrtUpdater(measurement_model)
    
    #Initialize Loop Variables
    track = Track()
    timeDiff = []
    for measurement in measurements:
        tic = time.perf_counter()
        prediction = predictor.predict(prior, timestamp=measurement.timestamp)
        hypothesis = SingleHypothesis(prediction, measurement)  # Group a prediction and measurement
        posterior = updater.update(hypothesis)
        toc = time.perf_counter()
        timeDiff.append(toc-tic)
        track.append(posterior)
        prior = track[-1]

    return track

def monte_carlo_runs(time_span, monte_carlo_iterations):
    transition_model, measurement_model = generate_models()
    try:
        with open('initial_state_list.py', 'rb') as f:
            initial_states = pickle.load(f)
    except:
        print('No initial states specified, generating now:')
        initial_states = generate_initial_states()
    EnKF_runs = []
    EnSRF_runs = []
    PCEnKF_runs = []
    PCEnSRF_runs = []
    ground_truth = []
    print('Monte Carlo runs starting:')
    for i in range(monte_carlo_iterations):
        tic = time.perf_counter()
        ground_truth.append(generate_ground_truth(transition_model,time_span))
        EnKF_runs.append(EnKF_Track(ground_truth[i], transition_model,measurement_model,initial_states[0]))
        EnSRF_runs.append(EnSRF_Track(ground_truth[i], transition_model,measurement_model,initial_states[1]))
        PCEnKF_runs.append(PCEnKF_Track(ground_truth[i], transition_model,measurement_model,initial_states[2]))
        PCEnSRF_runs.append(PCEnSRF_Track(ground_truth[i], transition_model,measurement_model,initial_states[3]))
        toc = time.perf_counter()
        print(f"Iteration number {i} complete in {toc - tic:0.4f} seconds")
 
    print('Monte Carlo Simulations Complete!')
    
    return EnKF_runs, EnSRF_runs, PCEnKF_runs, PCEnSRF_runs, ground_truth