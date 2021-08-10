import numpy as np
import pickle


def pack_RMSE_data(filename, datalist):
    with open(filename, 'wb') as f:
        pickle.dump([datalist], f)
          
def calc_std_error(data):
    """This function accepts a numpy array as input, and outputs an array of 
    size 2, containing the mean of all the elements in the first index, and 
    the standard error of the data in the second index."""
    return np.array([np.mean(data), np.std(data)/np.sqrt(len(data))])