import time
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.types.detection import Detection
from stonesoup.types.array import StateVector

from stonesoup.types.track import Track
from stonesoup.predictor.ensemble import EnsemblePredictor,PolynomialChaosEnsemblePredictor
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.updater.ensemble import (EnsembleUpdater, EnsembleSqrtUpdater,
             PolynomialChaosEnsembleUpdater, PolynomialChaosEnsembleSqrtUpdater)

from stonesoup.predictor.particle import ParticlePredictor
from stonesoup.resampler.particle import SystematicResampler
from stonesoup.updater.particle import ParticleUpdater

def generate_ground_truth(transition_model,time_span):
    time_interval = time_span[1]-time_span[0]
    truth = GroundTruthPath([GroundTruthState(StateVector([1000, 0, 2650, 150, 200, 0]),
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

    return Track, timeDiff

def EnKF_Track(ground_truth, transition_model, measurement_model, prior):
    #Simulate Measurements
    measurements = simulate_measurements(ground_truth,measurement_model)
    
    #Create Predictor and Updater
    predictor = EnsemblePredictor(transition_model)
    updater = EnsembleUpdater(measurement_model)
    
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

    return Track, timeDiff

def EnSRF_Track(ground_truth, transition_model, measurement_model, prior):
    #Simulate Measurements
    measurements = simulate_measurements(ground_truth,measurement_model)
    
    #Create Predictor and Updater
    predictor = EnsemblePredictor(transition_model)
    updater = EnsembleSqrtUpdater(measurement_model)
    
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

    return Track, timeDiff

def PCEnKF_Track(ground_truth, transition_model, measurement_model, prior):
    #Simulate Measurements
    measurements = simulate_measurements(ground_truth,measurement_model)
    
    #Create Predictor and Updater
    predictor = PolynomialChaosEnsemblePredictor(transition_model)
    updater = PolynomialChaosEnsembleUpdater(measurement_model)
    
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

    return Track, timeDiff

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

    return Track, timeDiff