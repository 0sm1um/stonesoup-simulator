import numpy as np
import matplotlib.pyplot as plt
from stonesoup.types.array import StateVectors

def calc_RMSE(ground_truth_list, track_list):
    """This function computes the RMSE of filter with respect to time.
    It accepts lists of ground truth paths, and tracks. It returns an instance
    of `StateVectors` in which columns contain Vectors of the RMSE at a given time"""
    if len(ground_truth_list) != len(track_list):
        print(len(ground_truth_list))
        print(len(track_list))
        return NotImplemented
    residual = np.zeros([ground_truth_list[0].states[0].ndim,len(ground_truth_list[0].states)])
    for instance in range(len(ground_truth_list)):
        ground_truth_states = StateVectors([e.state_vector for e in ground_truth_list[instance].states])
        tracked_states = StateVectors([e.state_vector for e in track_list[instance].states])
        residual = (tracked_states - ground_truth_states)**2 + residual
    RMSE = np.sqrt(residual/len(ground_truth_list))
    return RMSE

def plot_RMSE(RMSE,time_span):
    '''This function accepts an instance of `StateVectors` as an argument, and
    plots each of its root mean square errors as a function of time for each of
    its dimensions.  

    Parameters
    ----------
    RMSE : StateVectors
        DESCRIPTION.
    filter_name : string
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    x=RMSE[0]
    vx=RMSE[1]
    y=RMSE[2]
    vy=RMSE[3]
    t=time_span

    fig, ax = plt.subplots(ncols=2, nrows=3, constrained_layout=True)

    ax[0][0].plot(t, x)
    '''
    ax[0][0].title('$RMSE$ of {} versus $time$'.format(filter_name))
    ax[0][0].xlabel('$Time (in s)$')
    ax[0][0].ylabel('$RMSE of x$')
    '''
    ax[0][1].plot(t, vx)
    '''
    ax[1][0].title('$RMSE$ versus $time$')
    ax[1][0].xlabel('$Time (in s)$')
    ax[1][0].ylabel('$RMSE of vx$')
    '''
    ax[1][0].plot(t, y)
    '''
    ax[2][0].xlabel('$Time (in s)$')
    ax[2][0].ylabel('$RMSE of y$')
    '''
    ax[1][1].plot(t, vy)
    '''
    ax[0][1].xlabel('$Time (in s)$')
    ax[0][1].ylabel('$RMSE of vy$')
    '''
