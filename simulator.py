import numpy as np
import datetime

from stonesoup.types.array import StateVector, CovarianceMatrix
from stonesoup.types.state import GaussianState, EnsembleState

from stonesoup.models.transition.linear import (CombinedLinearGaussianTransitionModel,
                                                ConstantVelocity)
from stonesoup.models.transition.nonlinear import ConstantTurn

from stonesoup.models.measurement.nonlinear import (CartesianToElevationBearingRange, 
                                                    CartesianToBearingRange)
from stonesoup.models.measurement.linear import LinearGaussian


import numpy as np
import time

from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.types.detection import Detection
from stonesoup.types.array import StateVector, CovarianceMatrix

from stonesoup.types.track import Track
from stonesoup.predictor.kalman import KalmanPredictor, ExtendedKalmanPredictor
from stonesoup.predictor.ensemble import EnsemblePredictor
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.updater.ensemble import (EnsembleUpdater, EnsembleSqrtUpdater)
from stonesoup.updater.kalman import KalmanUpdater, ExtendedKalmanUpdater

from stonesoup.predictor.particle import ParticlePredictor
from stonesoup.resampler.particle import SystematicResampler
from stonesoup.updater.particle import ParticleUpdater



class simulator():
        
    def __init__(self, transition_model, measurement_model):
        """
        Parameters
        ----------
        transition_model : :class:`~.Predictor`
            The Stone Soup predictor to be used.
        measurement_model : :class:`~.Predictor`
            The Updater to be used.
        """
        self.transition_model= transition_model
        self.measurement_model = measurement_model        
    
    def _generate_ground_truth(self, prior, time_span):
        
        time_interval = time_span[1]-time_span[0]
        ground_truth = GroundTruthPath([prior])
        for k in range(1, len(time_span)):
            ground_truth.append(GroundTruthState(
                self.transition_model.function(ground_truth[k-1], noise=False, time_interval=time_interval),
                timestamp=time_span[k-1]))
        return ground_truth
    
    def _simulate_measurements(self, ground_truth):
        #Simulate Measurements
        measurements = []
        for state in ground_truth:
            measurement = self.measurement_model.function(state, noise=True)
            measurements.append(Detection(measurement,
                                          timestamp=state.timestamp,
                                          measurement_model=self.measurement_model))
        return measurements
    
    def simulate_track(self, predictor, updater, initial_state, prior, time_span):
        """

        Parameters
        ----------
        predictor : :class:`~.Predictor`
            The Stone Soup predictor to be used.
        updater : :class:`~.Predictor`
            The Updater to be used.
        ground_truth : :class:`~.GroundTruthPath`
            StateMutableSequence type object used to store ground truth.
        initial_state : :class:`~.State`
            Initial state for the ground truth system. This MUST be a State,
            not a State subclass, like GaussianState or EnsembleState.
        prior : :class:`~.GaussianState` or :class:`~.EnsembleState`
            Initial state prediction of tracking algorithm.

        Returns
        -------
        track : :class:`~.Track`
            The Stone Soup track object which contains the list of updated 
            state predictions made by the tracking algorithm employed.
        """
        
        #Simulate Measurements
        ground_truth = self._generate_ground_truth(initial_state, time_span)
        measurements = self._simulate_measurements(ground_truth)
        
        #Initialize Loop Variables
        track = Track()
        for measurement in measurements:
            prediction = predictor.predict(prior, timestamp=measurement.timestamp)
            hypothesis = SingleHypothesis(prediction, measurement)  # Group a prediction and measurement
            posterior = updater.update(hypothesis)
            track.append(posterior)
            prior = track[-1]
        return ground_truth, track
    