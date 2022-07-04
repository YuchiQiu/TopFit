import numpy as np
from scipy.stats import spearmanr

# Define mse loss
def mse(real, predicted):
    """
    Calculates mean squared error.

    Parameters
    ----------
    real: 1D numpy array
        The true values
    predicted: 1D numpy array
        The predicted values

    Returns
    -------
    mse: float
        Mean squared error
    """
    # Calculate the mse
    N = len(real)
    mse = (1 / N) * np.sum((real - predicted) ** 2)
    return mse
def spearman(y_pred, y_true):
    if np.var(y_pred) < 1e-6 or np.var(y_true) < 1e-6:
        return 0.0
    return spearmanr(y_pred, y_true).correlation

def cv_tuple(x,y,idx):
    """

    x: Tuple with two 2D numpy array or One 2D numpy array
        Features.
        Tuple: x[0] features from protein embedding
               x[1] features from unsupervised model
        2D numpy array: features for embedding only
    y: 1d numpy array
        Fitness vector
    idx: 1d numpy array
        index set for cross validation
    --------
    Return:
        x0: Tuple or 2D numpy array that is consistent with x.
            subset of x with index idx
        y0: 1d numpy array
            subset of y with index idx
    """
    if isinstance(x,tuple):
        assert len(x)==2;"tuple input must have two elements"
        x_0=x[0][idx]
        x_1=x[1][idx]
        x0=(x_0,x_1)
    else:
        x0=x[idx]
    y0=y[idx]
    return x0,y0

def merge_tuple(x,reg_coef=1.0):
    """

    x: Tuple with two 2D numpy array or One 2D numpy array
        Features.
        Tuple: x[0] features from protein embedding
               x[1] features from unsupervised model
        2D numpy array: features for embedding only
    reg_coef: Only useful when x is a Tuple
        Regression coefficient for unsupervised model.
        Default value is 1.0, the embedding and unsupervised features are treated equally
        It only takes non-default values for Ridge Regression.
    --------
    Return:
        x0: 2D numpy array
            If x is numpy array, x0 is equal to x
            If x is a Tuple, x0 append x[0] and x[1]
    """
    if isinstance(x,tuple):
        assert len(x)==2;"tuple input must have two elements"
        x0=np.append(x[0],x[1]/np.sqrt(reg_coef),axis=1)
    else:
        x0=x
    return x0

def device_tuple(x,device):
    if isinstance(x,tuple) or isinstance(x,list):
        assert len(x)==2;"tuple input must have two elements"
        return (x[0].to(device),x[1].to(device))
    else:
        return x.to(device)

def input_shape_tuple(x):
    if isinstance(x,tuple):
        assert len(x)==2;"tuple input must have two elements"
        return (x[0].shape[1],x[1].shape[1])
    else:
        return x.shape[1]

def single_structure_tuple(x,strucutre_id):
    """
    For features with multiple structures ONLY.
    Extract feature at `structure_id`-th structure.

    x: Tuple with two 3D numpy array or One 3D numpy array
        Features.
        Tuple: x[0] features from protein embedding
               x[1] features from unsupervised model
        3D numpy array: features for embedding only
        In either case, 3D numpy array has the shape NxCx L.
                N is number of sequences.
                C is number of structures considered for ensemble.
                    Usually used for NMR structure where multiple models are available
                L is the dimension of the encoding for each sequence
    structure_id: int
        The structure index to be extracted.
    --------
    Return:
    x0: Tuple with two 2D numpy array or One 2D numpy array. Depending on x
    """
    if isinstance(x,tuple):
        assert len(x)==2;"tuple input must have two elements"
        return (x[0][:,strucutre_id,:],x[1][:,strucutre_id,:])
    else:
        return x[:,strucutre_id,:]


