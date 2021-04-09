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
    RMSE = np.sqrt(residual/len(ground_truth_list[0].states))
    return RMSE

def plot_RMSE(RMSE_LIST,NAME_LIST):
    #Retrieve data from list
    for instance in range(len(RMSE_LIST)):
        x=[RMSE_LIST[instance][0]]
        vx=[RMSE_LIST[instance][1]]
        y=[RMSE_LIST[instance][2]]
        vy=[RMSE_LIST[instance][3]]
        z=[RMSE_LIST[instance][4]]
        vz=[RMSE_LIST[instance][5]]
    t=np.arange(len(RMSE_LIST[0][0]))
    plots = []
    
    
    
    for instance in range(len(RMSE_LIST)):
        fig = plt.figure
        plt.subplot(3, 2, 1)
        plt.plot(t, x[instance])
        plt.title('$RMSE$ versus $time$')
        plt.xlabel('$Time (in s)$')
        plt.ylabel('$RMSE of x$')
        
        plt.subplot(3, 2, 2)
        plt.plot(t, y[instance])
        plt.xlabel('$Time (in s)$')
        plt.ylabel('$RMSE of y$')
        
        plt.subplot(3, 2, 3)
        plt.plot(t, z[instance])
        plt.xlabel('$Time (in s)$')
        plt.ylabel('$RMSE of z$')
        
        plt.subplot(3, 2, 4)
        plt.plot(t, vx[instance])
        plt.title('$RMSE$ versus $time$')
        plt.xlabel('$Time (in s)$')
        plt.ylabel('$RMSE of vx$')
        
        plt.subplot(3, 2, 5)
        plt.plot(t, vy[instance])
        plt.xlabel('$Time (in s)$')
        plt.ylabel('$RMSE of vy$')
        
        plt.subplot(3, 2, 6)
        plt.plot(t, vz[instance])
        plt.xlabel('$Time (in s)$')
        plt.ylabel('$RMSE of vz$')
        
        plots.append(fig)

    return plots