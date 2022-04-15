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
from stonesoup.predictor.kalman import KalmanPredictor, ExtendedKalmanPredictor
from stonesoup.models.measurement.nonlinear import CartesianToBearingRange


from RMSE_functions import calc_RMSE, plot_RMSE
from MISC_functions import pack_RMSE_data

from simulator import simulator
"""
    This script is for Dave and I to play around with for our EGRE-656 Project
    
    Initially this will be a "tutorial" of sorts where I show you what I've 
    created and how to use it.
"""


"""
    In this first section, I instantiate a few variables we will need, 
    specifically relating to the time related parameters to our simulation.
    
    Note that the final line creates a vector "time_span". If you look at its 
    data type, it is a "numpy.ndarray" but the first element of it 
    time_span[0] is of type "datetime.datetime"
"""

#Timestamps are done in the native python datetime library.
#type "timestamp" and "type(timestamp)" in the spyder termiinal to see what
#the object looks like/exactly what data it stores.
#"datetime.datetime.now()" will also create a datetime object with the 
#current time on it

timestamp = datetime.datetime(2021, 4, 2, 12, 28, 57) #This is a datetime object
tMax = 120
dt = 1
tRange = tMax // dt #This is the length of the vector of datetime timestamps we want
time_span = np.array([timestamp + datetime.timedelta(seconds=dt*i) for i in range(tRange)])


"""
    Here we instantiate our transition and measurement models. These are 
    components native to the stonesoup library. If you check the top of this 
    script you can see where exactly in the filesystem they are located via 
    the import statements.
    
    These models are the exact same ones used in the Kalman Filter Tutorial.
"""

q_x = 0.05
q_y = 0.05

#Really basic 2D constant Velocity model.
#Pay attention to how this model is constructed. We have two one dimensional
#models called "ConstantVelocity" which are arguments for the 
#"CombinedLinearGaussianTransitionModel" object.
transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(q_x), 
                                                          ConstantVelocity(q_y)])
#PROTIP, if you type "LinearGaussian" in the spyder terminal and hit
# "ctrl+i" on your keyboard, it will pull up the documentation for that 
#component. This is true for any function or object with a defined Docstring.

measurement_model = LinearGaussian(
ndim_state=4,  # Number of state dimensions (position and velocity in 2D)
mapping=(0, 2),  # Mapping measurement vector index to state index
noise_covar=np.array([[0.05, 0],  # Covariance matrix for Gaussian PDF
                      [0, 0.05]])
)

"""
    Next, we instantiate the predictors we want to use. Again these will be 
    stonesoup components. For the tutorial's sake lets use the Kalman Filter.
    The KF is composed of a Predictor and an Updater. We need both of those.  
"""

#These simply accept the models we defined above as arguments.
#Check out the textbook's flow chart on the Kalman Filter to learn.
#Transition_Model is basically the "F" matrix from class.
#measurement model is the "H" matrix from class.

KFpredictor = KalmanPredictor(transition_model)
KFupdater = KalmanUpdater(measurement_model)

"""
    Lastly, as we will cover on class probably on monday, these algorithms 
    must be initialized to work. So lets set our prior state at t=0
"""
#This is object is a gaussian distribution with mean and covariance matrix
#specified below. The time is also set to the first element of our simulation
#time span. Play around with different priors and see what happens.

KFprior = GaussianState(state_vector=StateVector([0, 1, 0, 1]),
              covar = CovarianceMatrix(np.diag(np.array([0,0.05,0,0.05]))**2),
              timestamp = time_span[0])


"""
    Finally, we are prepared to use the simulator. The simulator is a tool 
    which isn't part of the stonesoup framework. It is a tool I wrote which 
    depends on stonesoup. 
    
    The premise is that we specify the transition and
    measurement model we wish to use. The simulator generates a set of points
    which is taken to be the true state of the system.
    
    This simulation is referred to as the "Ground Truth". It is the TRUE 
    system state which is known to us but not known to our tracking algorithm.
    
    Next, the simulator peturbs these ground truth points with statistical 
    noise. These peturbed points are our sequence of measurements. We then 
    feed these measurements into our tracking algorithm, in this case the 
    Kalman Filter.
    
    The Kalman Filter then runs, and generates a vector of estimates of what 
    it thinks is the state of the system. This is referred to as the "Track".
"""

simulator = simulator(transition_model=transition_model,
                      measurement_model=measurement_model)

#Note the way I've conceived this is that the simulator is an "object". Not
#a function. Think of it like a tool or a kitchen appliance. This particular 
#simulator has a linear gaussian transition and measurement model.

"""
Now, we will provide initial state for the ground truth, and the value we 
initialize our algorithms with. It is of course optional to initialize the 
algorithms with the same value as the ground truth, and I would actually 
encourage you not to do this.
"""

#Note its not Gaussian.
initial_ground_truth = State(state_vector=StateVector([0, 1, 0, 1]),
                             timestamp = time_span[0])

"""
Finally, we run our Simulations many times. Due to the randomness inherent to
stochastic simulation, we don't run a simulation once to evaluate an estimator.
We run it many many times, and average the Mean Square Error over the course
of many runs.
"""

#How many times do we want to run our simulation? Lets do 25!
monte_carlo_iterations = 25
KF_monte_carlo_runs = [] # Empty list to store simulation runs.

for i in range(monte_carlo_iterations):
    KF_monte_carlo_runs.append(simulator.simulate_track(predictor = KFpredictor, 
                                        updater = KFupdater, 
                                        initial_state = initial_ground_truth,
                                        prior = KFprior,
                                        time_span=time_span))
    print(monte_carlo_iterations) #This here is for YOUR benefit.

RMSE_KF = calc_RMSE(KF_monte_carlo_runs[0],KF_monte_carlo_runs[1])

"""
Finally here is a quick and dirty plot for you to see the averaged results
"""

plot_time_span = np.array([dt*i for i in range(tRange)])
plot_RMSE(RMSE_KF, plot_time_span)


