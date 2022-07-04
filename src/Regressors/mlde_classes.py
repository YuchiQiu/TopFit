"""
This file contains the classes which are the center of all MLDE calculations. On 
import, this module loads:

Classes
-------
MldeModel: Highest level class for performing MLDE operations
KerasModel: Container for keras models
XgbModel: Container for XGBoost models
SklearnRegressor: Container for sklearn regressor models
"""

# Filter convergence warnings
import warnings
import math
from math import sqrt
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)
# from multiprocessing import Pool
# from functools import partial

# Import sklearn regression objects
from sklearn.linear_model import (Ridge,ARDRegression, BayesianRidge, LassoLarsCV,
                                  SGDRegressor, ElasticNet, LinearRegression)
from sklearn.ensemble import (GradientBoostingRegressor, RandomForestRegressor,
                              BaggingRegressor, AdaBoostRegressor)
from sklearn.svm import LinearSVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

# Import all other packages
import numpy as np
import xgboost as xgb
from copy import deepcopy
import gc
import os

# Import MLDE packages
from utils import mse,spearman
from .input_check import check_training_inputs, check_keras_model_params

from .mlde_utils import cv_tuple,merge_tuple,device_tuple,input_shape_tuple

# Write a parent class that can train and predict using the smaller model classes
class MldeModel():
    """
    The main class for performing MLDE operations. Handles training and prediction
    for a given model architecture.
    
    Parameters
    ---------
    major_model: str
        Choice of 'Keras', 'XGB', 'sklearn-regressor', or 'Custom'. This argument
        tells MldeModel from which package we will be pulling models. 'Custom'
        allows the user to define their own class not directly in 'Keras', 
        'XGBoost', or 'sklearn'.
    specific_model: str
        This arguments tells MldeModel which regressor to use within the package
        defined by major_model. See online documentation for options.
    model_params: dict
        These are hyperparameters required for the construction of the models
        specified by 'major_model' and 'specific_model'. Details on the requirements
        for each model submodel can be found in the online documentation.
    training_params: dict
        These are parameters required for training the models specified by 
        'major_model' and 'specific_model'. Details on the requirements for each
        submodel can be found in the online documentation.
    eval_metric: func: default = mean squared error
        The function used for evaluating cross-validation error. This metric will
        be used to rank model architectures from best to worst. The function must
        take the form 'function(real_values, predicted_values)'.
    custom_model: class: default = None
        A user-defined class passed in when both 'major_model' and 'specific_model'
        are defined as 'Custom'. This model is designed to give greater flexibility
        to users of MLDE. Specific requirements for this custom class can be
        found on the online documentation.
    custom_model_args: iterable: default = []
        Iterable of arguments to pass in to 'custom_model'.
    custom_model_kwargs: dict: default = {}
        Dictionary of kwargs to pass in to 'custom_model'.
    
    Functions
    ---------
    self.train_cv(x, y, train_test_inds)
        Using the base model architecture given by 'self.major_model' and 
        'self.specific_model' trains MldeModel over a set of x and y values with
        k-fold cross validation. All models trained during cross validation are
        stored for use in generating predictions.
    self.predict(self, x):
        Returns the average predicted values for x over all models trained during
        train_cv. 
    self.clear_submodels():
        Force deletion of all trained models, and reset the Keras session. Keras
        sessions are not deleted unless this function is called. Trained models
        are deleted each time self.train_cv() is called.
        
    Attributes
    ----------
    self.major_model
    self.specific_model
    self.model_params
    self.training_params
    """
    # Initialize by defining the major and specific model type
    def __init__(self, major_model, specific_model, model_params = {}, 
                 training_params = {}, eval_metric = mse, 
                 custom_model = None, custom_model_args = [],
                 custom_model_kwargs = {},num_DNN=5):
        """
        Stores all input variables as instance variables. 'model_params' and 
        'training_params' are deep-copied prior to being stored as instance
        variables.

        num_DNN: number of repeats when using DNN models.
        self.predict() averages the predictions to gain robust results
        """
        # Store inputs as instance variables
        self._major_model = major_model
        self._specific_model = specific_model
        self._model_params = deepcopy(model_params)
        self._training_params = deepcopy(training_params)
        self._eval_metric = eval_metric
        self._custom_model = custom_model
        self._custom_model_args = custom_model_args
        self._custom_model_kwargs = custom_model_kwargs
        self._num_DNN=num_DNN
    # Define a function which builds a model according to the appropriate flavor
    def _build_model(self):
        """
        Private method which builds and returns the model specified by 
        'self.major_model' and 'self.specific_model', using the parameters given
        by 'self._model_params' and 'self._training_params'
        """
        # Confirm that the requested model is real
        assert self._major_model in _class_method_dict, f"Unknown major model: {self._major_model}"
        assert self._specific_model in _class_method_dict[self._major_model],\
            f"Unknown model: {self._major_model}-{self._specific_model}"
            
        # Return a generic model if that's what's requested
        if self._major_model == "Custom" and self._specific_model == "Custom":
            return self._custom_model(*self._custom_model_args, **self._custom_model_kwargs)
            
        # Construct and return the active model
        built_mod = _class_method_dict[self._major_model][self._specific_model]\
            (self._model_params, self._training_params)
        return built_mod
    
    # Define a function for training over a number of cross validation rounds
    def train_cv_multipleprocessing(self, x, y, train_test_inds, _debug = False):
        """
        Using the base model architecture given by 'self.major_model' and 
        'self.specific_model' trains MldeModel over a set of x and y values with
        k-fold cross validation. All models trained during cross validation are
        stored for use in generating predictions.
        
        Parameters
        ----------
        x:  Tuple with two 2D numpy array or One 2D numpy array
        Features.
            Tuple: x[0] features from protein embedding
               x[1] features from unsupervised model
            2D numpy array: features for embedding only
        y: 1D numpy arrayarray, length N
            Fitness values associated with each x
        train_test_inds: list of lists
            The cross-validation indices use in training.
        
        Returns
        -------
        training_loss: float
            Mean of cross-validation training error. Error is calculated using
            the function given by 'eval_metric' upon instantiation.        
        testing_loss: float
            Mean of cross-validation testing error. Error is calculated using
            the function given by 'eval_metric' upon instantiation.
        """
        if isinstance(x,tuple):
            x1=x[0]
            x2=x[1]
            assert len(x1) == len(x2),"Different number of training data in x1 and x2"
            assert len(x2) == len(y), "Different number of labels and training points"
        else:
            x2=x
            assert len(x2) == len(y), "Different number of labels and training points"
        
        # Identify cross validation indices and determine the number of models
        # needed.
        assert isinstance(train_test_inds, list), "Expected cross-validation indices to be a list"
        
        # Run checks on the cross-validation indices
        n_models_needed = len(train_test_inds)
        for train_inds, test_inds in train_test_inds:
            unique_train_inds = set(train_inds)
            unique_test_inds = set(test_inds)
            assert len(train_inds) + len(test_inds) == len(x2), "Cross val ind error"
            assert len(unique_train_inds) + len(unique_test_inds) == len(x2), "Duplicate cross val inds identified"
        
        # Initialize an instance variable for storing models
        self._models = [None for _ in range(n_models_needed)]

        # Initialize an array for storing loss
        training_loss = np.empty(n_models_needed)
        testing_loss = np.empty(n_models_needed)

        pool = Pool(processes=n_models_needed)
        partial_work = partial(self.train_cv_singlefold,x=x,y=y,
                               train_test_inds=train_test_inds)
        results = pool.map(partial_work, range(0, n_models_needed))
        pool.close()
        for i in range(n_models_needed):
            self._models[i]  = results[i][0]
            training_loss[i] = results[i][1]
            testing_loss[i] = results[i][2]

         # Return mean training and testing losses
        return training_loss.mean(), testing_loss.mean()


    def train_cv_singlefold(self,i,x,y,train_test_inds):
        # run one set of training and testing in cross-validation. Used for multiprocessing only.
        train_inds,test_inds=train_test_inds[i]
        # print(x[0].shape)
        # print(x[1].shape)
        # print(y.shape)
        # print(len(train_inds[0]))
        x_train, y_train = cv_tuple(x, y, train_inds)
        x_test, y_test = cv_tuple(x, y, test_inds)
        # Build a model
        active_mod = self._build_model()

        # Train the model
        active_mod.train(x_train, y_train, x_test, y_test)

        # Get predictions on training and testing data
        train_pred = active_mod.predict(x_train)
        test_pred = active_mod.predict(x_test)

        # Record the training and testing error
        training_loss = self._eval_metric(y_train, train_pred)
        testing_loss = self._eval_metric(y_test, test_pred)
        # Save the model object
        return active_mod,training_loss,testing_loss

    def train_cv(self, x, y, train_test_inds, _debug=False):
        """
        Using the base model architecture given by 'self.major_model' and
        'self.specific_model' trains MldeModel over a set of x and y values with
        k-fold cross validation. All models trained during cross validation are
        stored for use in generating predictions.

        Parameters
        ----------
        x: Tuple with two 2D numpy array or One 2D numpy array
        Features.
        Tuple: x[0] features from protein embedding
               x[1] features from unsupervised model
        2D numpy array: features for embedding only
        y: 1D numpy arrayarray, length N
            Fitness values associated with each x
        train_test_inds: list of lists
            The cross-validation indices use in training.

        Returns
        -------
        training_loss: float
            Mean of cross-validation training error. Error is calculated using
            the function given by 'eval_metric' upon instantiation.
        testing_loss: float
            Mean of cross-validation testing error. Error is calculated using
            the function given by 'eval_metric' upon instantiation.
        """
        if isinstance(x, tuple):
            x1 = x[0]
            x2 = x[1]
            assert len(x1) == len(x2), "Different number of training data in x1 and x2"
            assert len(x2) == len(y), "Different number of labels and training points"
        else:
            x2 = x
            assert len(x2) == len(y), "Different number of labels and training points"

        # Identify cross validation indices and determine the number of models
        # needed.
        assert isinstance(train_test_inds, list), "Expected cross-validation indices to be a list"

        # Run checks on the cross-validation indices
        n_models_needed = len(train_test_inds)
        for train_inds, test_inds in train_test_inds:
            unique_train_inds = set(train_inds)
            unique_test_inds = set(test_inds)
            assert len(train_inds) + len(test_inds) == len(x2), "Cross val ind error"
            assert len(unique_train_inds) + len(unique_test_inds) == len(x2), "Duplicate cross val inds identified"

        # Initialize an instance variable for storing models
        self._models = [None for _ in range(n_models_needed)]

        # Initialize an array for storing loss
        training_loss = np.empty(n_models_needed)
        testing_loss = np.empty(n_models_needed)

        # Loop over train-test inds
        for i, (train_inds, test_inds) in enumerate(train_test_inds):
            # Build x_train, x_test, y_train, and y_test
            x_train, y_train = cv_tuple(x, y, train_inds)
            x_test, y_test = cv_tuple(x, y, test_inds)
            # Build a model
            active_mod = self._build_model()

            # Train the model
            active_mod.train(x_train, y_train, x_test, y_test)

            # Get predictions on training and testing data
            train_pred = active_mod.predict(x_train)
            test_pred = active_mod.predict(x_test)

            # Record the training and testing error
            training_loss[i] = self._eval_metric(y_train, train_pred)
            testing_loss[i] = self._eval_metric(y_test, test_pred)

            # Save the model object
            self._models[i] = active_mod
        # Return mean training and testing losses
        return training_loss.mean(), testing_loss.mean()

    def train(self, x, y):
        """
        Using the base model architecture given by 'self.major_model' and
        'self.specific_model' trains MldeModel on x and y.

        Parameters
        ----------
        x: 2D numpy array, shape N x L. N is number of mutations, L is the length of featurization
        y: 1D numpy arrayarray, length N
            Fitness values associated with each x

        Returns
        -------
        training_loss: float
            Training error. Error is calculated using
            the function given by 'eval_metric' upon instantiation.
        """
        # Make sure that x and y are the same length
        # assert len(x) == len(y), "Different number of labels and training points"

        # Initialize an instance variable for storing models

        # Build a model
        # consider ensemble if the model is DNN. For robustness
        if self._major_model in ["DNN","DNN_deep"]:
            num_mods=self._num_DNN
        else:
            num_mods=1
        self._models=[None for _ in range(num_mods)]
        for i in range(num_mods):
            active_mod = self._build_model()
            # Train the model
            active_mod.train(x, y)
            self._models[i]=active_mod
            # pool = Pool(processes=num_mods)
            # partial_work = partial(self.train_single, x=x, y=y)
            # results = pool.map(partial_work, range(0, num_mods))
            # pool.close()
            # for i in range(num_mods):
            #     self._models[i] = results[i]
    def train_single(self,i,x,y):
        # function for single task serves for multiprocessing
        active_mod = self._build_model()
        active_mod.train(x, y)
        return active_mod
    # Write a function for predicting over a number of cross validation rounds
    # and (potentially) zero insertions
    def predict(self, x):
        """
        Returns the average predicted values for x over all models trained during
        train_cv. 
        
        Parameters
        ----------
        x: 2D or 3D numpy array, shape N x CL or N x C x L, respectively
            Array containing the encodings of the 'N' amino acid combinations to use
            for prediction. For all base models other than convolutional neural
            networks, input shape is N x CL (where 'C' is the number of amino acid
            positions bounding the combinatorial space and 'L' is the number of
            latent dimensions in the encoding). Convolutional neural networks
            expect a 3D array.
        
        Returns
        -------
        mean_preds: 1D numpy array: 
            The mean predicted labels over each model generated during training.
        stdev_preds: 1D numpy array:
            The standard deviation of predictions over each model generated
            during training.
        """
        # Create an array to store predictions in. Add an extra dimension if this
        predictions = []
        # Loop over the cross-validation models
        for i, model in enumerate(self._models):

            # Make and store predictions
            predictions.append(model.predict(x).flatten())
        predictions=np.asarray(predictions)
        # Get the mean and standard deviation of predictions
        mean_preds = np.mean(predictions, axis = 0)
        stdev_preds = np.std(predictions, axis = 0)
        # Return the mean predictions and standard deviation of predictions
        return mean_preds, stdev_preds

    # Write a function that clears all cached models session
    def clear_submodels(self):
        """
        Force deletion of all trained models, and reset the Keras session. Keras
        sessions are not deleted unless this function is called. Trained models
        are deleted each time self.train_cv() is called.
        """
        # If a keras model, clear it
        if self._major_model == "Keras":
            
            # If we have the attribute "_models", delete all
            for model in self._models:
                model.clear_model()
                                                   
        # Delete all active models
        for model in self._models:
            del(model)
            
        # Delete the model and saved_model lists
        del(self._models)
        
        # Collect garbage
        _ = gc.collect()            

    # Set properties of all models people might want to call
    @property
    def major_model(self):
        return self._major_model
    
    @property
    def specific_model(self):
        return self._specific_model
    
    @property
    def model_params(self):
        return self._model_params
    
    @property
    def training_params(self):
        return self._training_params


# Write a class for the XgbModel
class XgbModel():
    """
    Container for all models built in XGBoost. A number of models are already
    attached to this class, including XgbModel.Linear, XgbModel.Tree, 
    XgbModel.LinearTweedie, and XgbModel.TreeTweedie. 
    
    Parameters
    ----------
    model_params: dict
        These are the parameters passed to xgb.train(), and define the architecture
        of the XGBoost model. See the XGBoost docs for more info on the 'param'
        argument passed in to xgb.train()
    training_params: dict
        These are all optional keyword arguments passed in to xgb.train(). 
        
    Functions
    ---------
    self.train(x_train, y_train, x_test, y_test):
        Trains the input XgbModel with early stopping, using the model defined
        by 'model_params' and training keyword found in 'training_params'.
        Early stopping is employed.
    self.predict(x):
        Generates predicted labels for 'x' based on the model trained in self.train()
    XgbModel.Linear(model_params, training_params):
        Generates an instance of XgbModel using an XGBoost model with a base
        linear model. Standard regression is used for this model. 
    XgbModel.Tree(model_params, training_params):
        Generates an instance of XgbModel using an XGBoost model with a base
        tree model. Standard regression is used for this model. 
    XgbModel.LinearTweedie(model_params, training_params):
        Generates an instance of XgbModel using an XGBoost model with a base
        linear model. Tweedie regression is used for this model.
    XgbModel.TreeTweedie(model_params, training_params):
        Generates an instance of XgbModel using an XGBoost model with a base
        tree model. Tweedie regression is used for this model.        
        
    Attributes
    ----------
    self.early_stopping_epoch
    self.training_params
    self.model_params
    """
    # Define an initilization function which sets up all model parameters
    def __init__(self, model_params, training_params):
        """
        Copies 'model_params' and 'training_params' and stores them as intance
        variables.
        """
        # Set model and training parameters as instance variables


        self._model_params = deepcopy(model_params)
        self._training_params = deepcopy(training_params)
    # Define a function for training the model on one set of x and y
    def train(self, x_train, y_train, x_test=None, y_test=None):
        """
        Trains the input XgbModel with early stopping, using the model defined
        by 'model_params' and training keyword found in 'training_params'.
        Early stopping is employed.
        
        Parameters
        ----------
        x_train: 2D numpy array, shape N x CL
            Array containing the encodings of the 'N' amino acid combinations to use
            for training.'C' is the number of amino acid positions bounding the
            combinatorial space and 'L' is the number of latent dimensions in
            the encoding
        y_train: 1D numpy array
            Labels to use in training
        x_test: 2D or 3D numpy array
            x-values to use in calcuating test error. 
        y_test: 1D numpy array
            Labels to use in calculating test error.
        """
        # Assert that x is 2D
        # assert len(x_train.shape) == 2, "x values must be a 2D matrix"
        x_train=merge_tuple(x_train)
        # Make generic checks on input
        # if x_test is not None and y_test is not None:
        #     assert len(x_test.shape) == 2, "x values must be a 2D matrix"
        #     check_training_inputs(x_train, y_train, x_test, y_test)
        #     test_matrix = xgb.DMatrix(x_test, label=y_test)
        # Build DMatrices
        train_matrix = xgb.DMatrix(x_train, label = y_train)


        # Create an eval list
        # evallist = [(train_matrix, "train"),
        #             (test_matrix, "test")]

        # Train the model and store as the "mod" variable
        self._mod = xgb.train(self._model_params, train_matrix,
                              # evals = evallist,
                              **self._training_params)

        # Identify the early stopping epoch
        # self._early_stopping_epoch = self._mod.best_ntree_limit

    # Define a function for predicting from a single model instance
    def predict(self, x):
        """
        Generates predicted labels for 'x' based on the model trained in self.train()
        
        Parameters
        ----------
        x: 2D numpy array, shape N x CL
            Array containing the encodings of the 'N' amino acid combinations for
            which to predict labels.'C' is the number of amino acid positions 
            bounding the combinatorial space and 'L' is the number of latent
            dimensions in the encoding
            
        Returns
        -------
        preds: 1D numpy array
            Predicted labels for 'x'
        """      
        # Assert that x is 2d
        # assert len(x.shape) == 2, "Expected a 2D input for x"
        x=merge_tuple(x)

        # Return predicted values (don't use best iteration if linear, because
        # there is no supported way in xgboost to do this currently...)

        # if self._model_params["booster"] == "gblinear":
        preds = self._mod.predict(xgb.DMatrix(x))
        # else:
        #     preds = self._mod.predict(xgb.DMatrix(x),
        #                              ntree_limit = self._early_stopping_epoch)
                
        # Return predictions
        return preds.flatten()
        
    # Set properties
    @property
    def early_stopping_epoch(self):
        return self._early_stopping_epoch
    
    @property
    def training_params(self):
        return self._training_params
    
    @property
    def model_params(self):
        return self._model_params
    
    @property
    def mod(self):
        return self._mod
    
    # Define a class method for building a linear booster
    @classmethod
    def Linear(cls, model_params, training_params):
        """
        Generates an instance of XgbModel using an XGBoost model with a base
        linear model. Standard regression is used for this model. 
        
        Parameters
        ----------
        model_params: dict
            Additional parameters to use when building a linear XGBoost model. 
            See XGBoost docs for details.
        training_params: dict
            Passed directly to XgbModel
            
        Returns
        -------
        An instance of XgbModel using an XGBoost model with a base linear model,
        using 'reg:squarederror' as the XGBoost regression objective and 'rmse'
        as the XGBoost eval metric.
        """
        # Build general model parameters
        mod_params = {"booster": "gblinear",
                      "tree_method": "exact",
                      "nthread": 1,
                      "verbosity": 0,
                      "objective": "reg:squarederror",
                      "eval_metric": "rmse"}

        # Add specific model parameters
        mod_params.update(model_params)

        # Create an instance
        return cls(mod_params, training_params)

    # Define a class method for building a tree booster
    @classmethod
    def Tree(cls, model_params, training_params):
        """
        Generates an instance of XgbModel using an XGBoost model with a base
        tree model. Standard regression is used for this model. 
        
        Parameters
        ----------
        model_params: dict
            Additional parameters to use when building a tree XGBoost model. 
            See XGBoost docs for details.
        training_params: dict
            Passed directly to XgbModel
            
        Returns
        -------
        An instance of XgbModel using an XGBoost model with a base tree model,
        using 'reg:squarederror' as the XGBoost regression objective and 'rmse'
        as the XGBoost eval metric.
        """
        # Set model parameters
        mod_params = {"booster": "gbtree",
                      "tree_method": "exact",
                      "nthread": 1,
                      "verbosity": 0,
                      "objective": "reg:squarederror",
                      "eval_metric": "rmse"}

        # Add specific model parameters
        mod_params.update(model_params)

        # Create an instance
        return cls(mod_params, training_params)

    # Define a class method for building a Tweedie linear booster
    @classmethod
    def LinearTweedie(cls, model_params, training_params):
        """
        Generates an instance of XgbModel using an XGBoost model with a base
        linear model. Tweedie regression is used for this model.
        
        Parameters
        ----------
        model_params: dict
            Additional parameters to use when building a linear XGBoost model
            with the tweedie regression objective. See XGBoost docs for details.
        training_params: dict
            Passed directly to XgbModel
            
        Returns
        -------
        An instance of XgbModel using an XGBoost model with a base linear model,
        using 'reg:tweedie' as the XGBoost regression objective, 
        'tweedie_variance_power' or 1.5, and 'tweedie-nloglik@1.5' as the
        XGBoost eval metric.
        """
        # Build general model parameters
        mod_params = {"booster": "gblinear",
                      "tree_method": "exact",
                      "nthread": 1,
                      "verbosity": 0,
                      "objective": "reg:tweedie",
                      "tweedie_variance_power": 1.5,
                      "eval_metric": "tweedie-nloglik@1.5"}

        # Add specific model parameters
        mod_params.update(model_params)

        # Create an instance
        return cls(mod_params, training_params)

    # Define a class method for building a Tweedie tree booster
    @classmethod
    def TreeTweedie(cls, model_params, training_params):
        """
        Generates an instance of XgbModel using an XGBoost model with a base
        tree model. Tweedie regression is used for this model.
        
        Parameters
        ----------
        model_params: dict
            Additional parameters to use when building a tree XGBoost model
            with the tweedie regression objective. See XGBoost docs for details.
        training_params: dict
            Passed directly to XgbModel
            
        Returns
        -------
        An instance of XgbModel using an XGBoost model with a base tree model,
        using 'reg:tweedie' as the XGBoost regression objective, 
        'tweedie_variance_power' or 1.5, and 'tweedie-nloglik@1.5' as the
        XGBoost eval metric.
        """
        # Set model parameters
        mod_params = {"booster": "gbtree",
                      "tree_method": "exact",
                      "nthread": 1,
                      "verbosity": 0,
                      "objective": "reg:tweedie",
                      "tweedie_variance_power": 1.5,
                      "eval_metric": "tweedie-nloglik@1.5"}

        # Add specific model parameters
        mod_params.update(model_params)

        # Create an instance
        return cls(mod_params, training_params)

# Write a class for the SklearnRegressor
class SklearnRegressor():
    """
    Container for sklearn models. 
    
    Parameters
    ----------
    mod: sklearn regressor object
        A regressor object from the scikit-learn machine learning module
    placeholder: dummy variable: default = None
        This variable is not used. It is in place solely to keep this container
        compatible with KerasModel and XgbModel
        
    Functions
    ---------
    self.train(x_train, y_train, x_test, y_test):
        Trains the input sklearn model
    self.predict(x):
        Generates predicted labels for 'x' based on the model trained in self.train()
    SklearnRegressor.Linear(model_params):
        Generates a SklearnRegressor instance using the LinearRegression sklearn
        model.
    SklearnRegressor.GradientBoostingRegressor(model_params):
        Generates a SklearnRegressor instance using the GradientBoostingRegressor sklearn
        model.
    SklearnRegressor.RandomForestRegressor(model_params):
        Generates a SklearnRegressor instance using the RandomForestRegressor sklearn
        model.
    SklearnRegressor.LinearSVR(model_params):
        Generates a SklearnRegressor instance using the LinearSVR sklearn
        model.
    SklearnRegressor.ARDRegression(model_params):
        Generates a SklearnRegressor instance using the ARDRegression sklearn
        model.
    SklearnRegressor.KernelRidge(model_params):
        Generates a SklearnRegressor instance using the KernelRidge sklearn
        model.
    SklearnRegressor.BayesianRidge(model_params):
        Generates a SklearnRegressor instance using the BayesianRidge sklearn
        model.
    SklearnRegressor.BaggingRegressor(model_params):
        Generates a SklearnRegressor instance using the BaggingRegressor sklearn
        model.
    SklearnRegressor.LassoLarsCV(model_params):
        Generates a SklearnRegressor instance using the LassoLarsCV sklearn
        model.
    SklearnRegressor.DecisionTreeRegressor(model_params):
        Generates a SklearnRegressor instance using the DecisionTreeRegressor sklearn
        model.
    SklearnRegressor.SGDRegressor(model_params):
        Generates a SklearnRegressor instance using the SGDRegressor sklearn
        model.
    SklearnRegressor.KNeighborsRegressor(model_params):
        Generates a SklearnRegressor instance using the KNeighborsRegressor sklearn
        model.
    SklearnRegressor.ElasticNet(model_params):
        Generates a SklearnRegressor instance using the ElasticNet sklearn
        model.
    Attributes
    ----------
    self.mod
    """
    # Define an initilization function which sets up all model parameters
    def __init__(self, mod, placeholder = None): 
        """
        Assigns 'mod' as an instance variable.
        """
        # Set the model as an instance variable as well as the model parameters
        self._mod = mod

    # Define a function for training the model on one set of x and y
    def train(self, x_train, y_train, x_test=None, y_test=None):
        """
        Trains the input sklearn model
        
        Parameters
        ----------
        x_train: 2D numpy array, shape N x CL
            Array containing the encodings of the 'N' amino acid combinations to use
            for training.'C' is the number of amino acid positions bounding the
            combinatorial space and 'L' is the number of latent dimensions in
            the encoding
        y_train: 1D numpy array
            Labels to use in training
        x_test: 2D or 3D numpy array
            x-values to use in calcuating test error. 
        y_test: 1D numpy array
            Labels to use in calculating test error.
        """
        # Test the input data
        # check_training_inputs(x_train, y_train, x_test, y_test)
        
        # Make sure x is 2D
        # assert len(x_train.shape) == 2, "x values must be a 2D matrix"
        # if x_test is not None:
        #     assert len(x_test.shape) == 2, "x values must be a 2D matrix"

        # If the model is LinearRegressor, reshape y_train to be 2d
        if self._mod.__class__.__name__ == "LinearRegression":
            y_train = np.expand_dims(y_train, axis = 1)

        # Give a small l2 regularization for Unsupervised features if Ridge regression is used
        if self._mod.__class__.__name__ in ["Ridge","KernelRidge"]:
            self.reg_coef=10**-8
        else:
            self.reg_coef=1.0

        x_train=merge_tuple(x_train,reg_coef=self.reg_coef)

        # Fit the model
        self._mod.fit(x_train, y_train)

    # Define a function for predicting from a single model instance
    def predict(self, x):
        """
        Generates predicted labels for 'x' based on the model trained in self.train()
        
        Parameters
        ----------
        x: 2D numpy array, shape N x CL
            Array containing the encodings of the 'N' amino acid combinations for
            which to predict labels.'C' is the number of amino acid positions 
            bounding the combinatorial space and 'L' is the number of latent
            dimensions in the encoding
            
        Returns
        -------
        preds: 1D numpy array
            Predicted labels for 'x'
        """      
        # Throw an error if x is not 2D
        # assert len(x.shape) == 2, "x must be 2D"
        x=merge_tuple(x,reg_coef=self.reg_coef)

        # Return the prediction
        return self._mod.predict(x).flatten()
    
    # Create properties
    @property
    def mod(self):
        return self._mod

    # Define a class method for building a linear regressor
    @classmethod
    def Linear(cls, model_params, training_params = None):
        """
        Generates a SklearnRegressor instance using the LinearRegression sklearn
        model.
        
        Parameters
        ----------
        model_params: dict
            Kwargs passed in to sklearn's LinearRegression class
        training_params: None
            Unused. Present to remain compatible with other containers.
        """
        # Build the sklearn model instance
        mod = LinearRegression(**model_params)

        # Construct with the initializer
        return cls(mod)

    # Define a class method for building with a gradient boosting regressor
    @classmethod
    def GradientBoostingRegressor(cls, model_params, training_params = None):
        """
        Generates a SklearnRegressor instance using the GradientBoostingRegressor sklearn
        model.
        
        Parameters
        ----------
        model_params: dict
            Kwargs passed in to sklearn's GradientBoostingRegressor class
        training_params: None
            Unused. Present to remain compatible with other containers.
        """
        # Build the sklearn instance
        mod = GradientBoostingRegressor(**model_params)

        # Return an instance
        return cls(mod)

    # Define a class method for building with a RandomForestRegressor
    @classmethod
    def RandomForestRegressor(cls, model_params, training_params = None):
        """
        Generates a SklearnRegressor instance using the RandomForestRegressor sklearn
        model.
        
        Parameters
        ----------
        model_params: dict
            Kwargs passed in to sklearn's RandomForestRegressor class
        training_params: None
            Unused. Present to remain compatible with other containers.
        """
        # Build the sklearn instance
        mod = RandomForestRegressor(**model_params)

        # Create an instance
        return cls(mod)

    # Define a class method for building LinearSVR
    @classmethod
    def LinearSVR(cls, model_params, training_params = None):
        """
        Generates a SklearnRegressor instance using the LinearSVR sklearn
        model.
        
        Parameters
        ----------
        model_params: dict
            Kwargs passed in to sklearn's LinearSVR class
        training_params: None
            Unused. Present to remain compatible with other containers.
        """
        # Build the sklearn instance
        mod = LinearSVR(**model_params)

        # Return an instance
        return cls(mod)

    # Define a class method for ARDRegression
    @classmethod
    def ARDRegression(cls, model_params, training_params = None):
        """
        Generates a SklearnRegressor instance using the ARDRegression sklearn
        model.
        
        Parameters
        ----------
        model_params: dict
            Kwargs passed in to sklearn's ARDRegression class
        training_params: None
            Unused. Present to remain compatible with other containers.
        """
        # Build the sklearn instance
        mod = ARDRegression(**model_params)

        # Return an instance
        return cls(mod)

    # Define a class method for KernelRidge
    @classmethod
    def KernelRidge(cls, model_params, training_params = None):
        """
        Generates a SklearnRegressor instance using the KernelRidge sklearn
        model.
        
        Parameters
        ----------
        model_params: dict
            Kwargs passed in to sklearn's KernelRidge class
        training_params: None
            Unused. Present to remain compatible with other containers.
        """
        # Build the sklearn instance
        mod = KernelRidge(**model_params)

        # Return an instance
        return cls(mod)

    @classmethod
    def RidgeRegression(cls, model_params, training_params=None):
        """
        Generates a SklearnRegressor instance using the Ridge sklearn
        model.

        Parameters
        ----------
        model_params: dict
            Kwargs passed in to sklearn's KernelRidge class
        training_params: None
            Unused. Present to remain compatible with other containers.
        """
        # Build the sklearn instance
        mod = Ridge(**model_params)

        # Return an instance
        return cls(mod)
    # Define a class method for BayesianRidge
    @classmethod
    def BayesianRidge(cls, model_params, training_params = None):
        """
        Generates a SklearnRegressor instance using the BayesianRidge sklearn
        model.
        
        Parameters
        ----------
        model_params: dict
            Kwargs passed in to sklearn's BayesianRidge class
        training_params: None
            Unused. Present to remain compatible with other containers.
        """
        # Build the sklearn instance
        mod = BayesianRidge(**model_params)

        # Return an instance
        return cls(mod)

    # Define a class method for BaggingRegressor
    @classmethod
    def BaggingRegressor(cls, model_params, training_params = None):
        """
        Generates a SklearnRegressor instance using the BaggingRegressor sklearn
        model.
        
        Parameters
        ----------
        model_params: dict
            Kwargs passed in to sklearn's BaggingRegressor class
        training_params: None
            Unused. Present to remain compatible with other containers.
        """
        # Build the sklearn instance
        mod = BaggingRegressor(**model_params)

        # Return an instance
        return cls(mod)

    # Define a class method for LassoLarsCV
    @classmethod
    def LassoLarsCV(cls, model_params, training_params = None):
        """
        Generates a SklearnRegressor instance using the LassoLarsCV sklearn
        model.
        
        Parameters
        ----------
        model_params: dict
            Kwargs passed in to sklearn's LassoLarsCV class
        training_params: None
            Unused. Present to remain compatible with other containers.
        """
        # Build the sklearn instance
        mod = LassoLarsCV(**model_params)

        # Return an instance
        return cls(mod)

    # Define a classmethod for DecisionTreeRegressor
    @classmethod
    def DecisionTreeRegressor(cls, model_params, training_params = None):
        """
        Generates a SklearnRegressor instance using the DecisionTreeRegressor sklearn
        model.
        
        Parameters
        ----------
        model_params: dict
            Kwargs passed in to sklearn's DecisionTreeRegressor class
        training_params: None
            Unused. Present to remain compatible with other containers.
        """
        # Build the sklearn instance
        mod = DecisionTreeRegressor(**model_params)

        # Return an instance
        return cls(mod)

    # Define a classmethod for SGDRegressor
    @classmethod
    def SGDRegressor(cls, model_params, training_params = None):
        """
        Generates a SklearnRegressor instance using the SGDRegressor sklearn
        model.
        
        Parameters
        ----------
        model_params: dict
            Kwargs passed in to sklearn's SGDRegressor class
        training_params: None
            Unused. Present to remain compatible with other containers.
        """
        # Build the sklearn instance
        mod = SGDRegressor(**model_params)

        # Return an instance
        return cls(mod)

    # Define a classmethod for KNeighborsRegressor
    @classmethod
    def KNeighborsRegressor(cls, model_params, training_params = None):
        """
        Generates a SklearnRegressor instance using the KNeighborsRegressor sklearn
        model.
        
        Parameters
        ----------
        model_params: dict
            Kwargs passed in to sklearn's KNeighborsRegressor class
        training_params: None
            Unused. Present to remain compatible with other containers.
        """
        # Build the sklearn instance
        mod = KNeighborsRegressor(**model_params)

        # Return an instance
        return cls(mod)

    # Define a classmethod for ElasticNet
    @classmethod
    def ElasticNet(cls, model_params, training_params = None):
        """
        Generates a SklearnRegressor instance using the ElasticNet sklearn
        model.
        
        Parameters
        ----------
        model_params: dict
            Kwargs passed in to sklearn's ElasticNet class
        training_params: None
            Unused. Present to remain compatible with other containers.
        """
        # Build the sklearn instance
        mod = ElasticNet(**model_params)

        # Return an instance
        return cls(mod)


class DNNRegressor():
    # Define an initilization function which sets up all model parameters
    def __init__(self, mod, training_params):
        self._training_params = deepcopy(training_params)
        self._mod = mod

    def train(self, x_train, y_train, x_test=None, y_test=None):
        """
        Trains the input sklearn model

        Parameters
        ----------
        x_train: 2D numpy array, shape N x CL
            Array containing the encodings of the 'N' amino acid combinations to use
            for training.'C' is the number of amino acid positions bounding the
            combinatorial space and 'L' is the number of latent dimensions in
            the encoding
        y_train: 1D numpy array
            Labels to use in training
        x_test: 2D or 3D numpy array
            x-values to use in calcuating test error.
        y_test: 1D numpy array
            Labels to use in calculating test error.
        """
        # Test the input data
        # check_training_inputs(x_train, y_train, x_test, y_test)

        # Make sure x is 2D
        # assert len(x_train.shape) == 2, "x values must be a 2D matrix"
        # if x_test is not None:
        #     assert len(x_test.shape) == 2, "x values must be a 2D matrix"
        # Fit the model
        self._mod.fit(x_train, y_train)

    def predict(self, x):
        """
        Generates predicted labels for 'x' based on the model trained in self.train()

        Parameters
        ----------
        x: 2D numpy array, shape N x CL
            Array containing the encodings of the 'N' amino acid combinations for
            which to predict labels.'C' is the number of amino acid positions
            bounding the combinatorial space and 'L' is the number of latent
            dimensions in the encoding

        Returns
        -------
        preds: 1D numpy array
            Predicted labels for 'x'
        """
        # Throw an error if x is not 2D
        # x = merge_tuple(x)

        # Return the prediction
        return self._mod.predict(x).flatten()

    @property
    def mod(self):
        return self._mod
    #Define a class method that makes a model with one hidden layers
    @classmethod
    def OneHidden(cls, model_params, training_params):
        model_params.update(training_params)
        model_params['hiddens']=[model_params['size1']]
        assert len(model_params['hiddens'])==1; "Expected to have a 1D list for one layer size"
        # del model_params['size1']
        mod=DNN(**model_params)
        return cls(mod,training_params)
    @classmethod
    def TwoHidden(cls, model_params, training_params):
        model_params.update(training_params)
        model_params['hiddens']=[model_params['size1'],model_params['size2']]
        assert len(model_params['hiddens'])==2; "Expected to have a 1D list for one layer size"
        # del model_params['size1']
        # del model_params['size2']
        mod=DNN(**model_params)
        return cls(mod,training_params)
    @classmethod
    def ThreeHidden(cls, model_params, training_params):
        model_params.update(training_params)
        model_params['hiddens']=[model_params['size1'],model_params['size2'],model_params['size3']]
        assert len(model_params['hiddens'])==3; "Expected to have a 1D list for one layer size"
        # del model_params['size1']
        # del model_params['size2']
        mod=DNN(**model_params)
        return cls(mod,training_params)

class DNNDeepRegressor():
    # Define an initilization function which sets up all model parameters
    def __init__(self, mod, training_params):
        self._training_params = deepcopy(training_params)
        self._mod = mod

    def train(self, x_train, y_train, x_test=None, y_test=None):
        """
        Trains the input sklearn model

        Parameters
        ----------
        x_train: 2D numpy array, shape N x CL
            Array containing the encodings of the 'N' amino acid combinations to use
            for training.'C' is the number of amino acid positions bounding the
            combinatorial space and 'L' is the number of latent dimensions in
            the encoding
        y_train: 1D numpy array
            Labels to use in training
        x_test: 2D or 3D numpy array
            x-values to use in calcuating test error.
        y_test: 1D numpy array
            Labels to use in calculating test error.
        """
        # Test the input data
        # check_training_inputs(x_train, y_train, x_test, y_test)

        # Make sure x is 2D
        # assert len(x_train.shape) == 2, "x values must be a 2D matrix"
        # if x_test is not None:
        #     assert len(x_test.shape) == 2, "x values must be a 2D matrix"
        # Fit the model
        self._mod.fit(x_train, y_train)

    def predict(self, x):
        """
        Generates predicted labels for 'x' based on the model trained in self.train()

        Parameters
        ----------
        x: 2D numpy array, shape N x CL
            Array containing the encodings of the 'N' amino acid combinations for
            which to predict labels.'C' is the number of amino acid positions
            bounding the combinatorial space and 'L' is the number of latent
            dimensions in the encoding

        Returns
        -------
        preds: 1D numpy array
            Predicted labels for 'x'
        """
        # Throw an error if x is not 2D
        # x = merge_tuple(x)

        # Return the prediction
        return self._mod.predict(x).flatten()

    @property
    def mod(self):
        return self._mod
    #Define a class method that makes a model with one hidden layers
    @classmethod
    def OneHidden(cls, model_params, training_params):
        model_params.update(training_params)
        model_params['hiddens']=[model_params['size1']]
        assert len(model_params['hiddens'])==1; "Expected to have a 1D list for one layer size"
        # del model_params['size1']
        mod=DNN(**model_params)
        return cls(mod,training_params)
    @classmethod
    def TwoHidden(cls, model_params, training_params):
        model_params.update(training_params)
        model_params['hiddens']=[model_params['size1'],model_params['size2']]
        assert len(model_params['hiddens'])==2; "Expected to have a 1D list for one layer size"
        # del model_params['size1']
        # del model_params['size2']
        mod=DNN(**model_params)
        return cls(mod,training_params)
    @classmethod
    def ThreeHidden(cls, model_params, training_params):
        model_params.update(training_params)
        model_params['hiddens']=[model_params['size1'],model_params['size2'],model_params['size3']]
        assert len(model_params['hiddens'])==3; "Expected to have a 1D list for one layer size"
        # del model_params['size1']
        # del model_params['size2']
        mod=DNN(**model_params)
        return cls(mod,training_params)
    @classmethod
    def FourHidden(cls, model_params, training_params):
        model_params.update(training_params)
        model_params['hiddens']=[model_params['size1'],model_params['size2'],model_params['size3'],model_params['size4']]
        assert len(model_params['hiddens'])==4; "Expected to have a 1D list for one layer size"
        # del model_params['size1']
        # del model_params['size2']
        mod=DNN(**model_params)
        return cls(mod,training_params)
    @classmethod
    def FiveHidden(cls, model_params, training_params):
        model_params.update(training_params)
        model_params['hiddens']=[model_params['size1'],model_params['size2'],model_params['size3'],model_params['size4'],model_params['size5']]
        assert len(model_params['hiddens'])==5; "Expected to have a 1D list for one layer size"
        # del model_params['size1']
        # del model_params['size2']
        mod=DNN(**model_params)
        return cls(mod,training_params)

# Define a dictionary structure for calling class methods
_class_method_dict = {"XGB": {"Tree": XgbModel.Tree,
                             "Linear": XgbModel.Linear,
                             "Tree-Tweedie": XgbModel.TreeTweedie,
                             "Linear-Tweedie": XgbModel.LinearTweedie,
                             "Custom": XgbModel},
                     "sklearn-regressor": {"Linear": SklearnRegressor.Linear,
                                           "GradientBoostingRegressor": SklearnRegressor.GradientBoostingRegressor,
                                           "RandomForestRegressor": SklearnRegressor.RandomForestRegressor,
                                           "LinearSVR": SklearnRegressor.LinearSVR,
                                           "ARDRegression": SklearnRegressor.ARDRegression,
                                           "KernelRidge": SklearnRegressor.KernelRidge,
                                           "BayesianRidge": SklearnRegressor.BayesianRidge,
                                           "BaggingRegressor": SklearnRegressor.BaggingRegressor,
                                           "LassoLarsCV": SklearnRegressor.LassoLarsCV,
                                           "DecisionTreeRegressor": SklearnRegressor.DecisionTreeRegressor,
                                           "SGDRegressor": SklearnRegressor.SGDRegressor,
                                           "KNeighborsRegressor": SklearnRegressor.KNeighborsRegressor,
                                           "ElasticNet": SklearnRegressor.ElasticNet,
                                           "Ridge":SklearnRegressor.RidgeRegression,
                                           "Custom": SklearnRegressor},
                      "DNN":{'OneHidden':DNNRegressor.OneHidden,
                             'TwoHidden':DNNRegressor.TwoHidden,
                              'ThreeHidden':DNNRegressor.ThreeHidden,
                             },
                      "DNN_deep":{'OneHidden':DNNDeepRegressor.OneHidden,
                                  'TwoHidden':DNNDeepRegressor.TwoHidden,
                                  'ThreeHidden':DNNDeepRegressor.ThreeHidden,
                                  'FourHidden':DNNDeepRegressor.FourHidden,
                                  'FiveHidden':DNNDeepRegressor.FiveHidden,
                                  },
                     "Custom": {"Custom"}
}
class Augmented_Dataset(Dataset):
    def __init__(self, X, y=None, transforms=transforms.Compose([])):
        self.X=X
        if y is None:
            self.labels = None
        else:
            self.labels=y
        self.transforms = transforms

    def __getitem__(self, index):
        if isinstance(self.X,tuple):
            assert len(self.X)==2; "tuple input must have two elements"
            X_embedding_tensor = torch.from_numpy(self.X[0][index]).float()
            X_unsupervised_tensor = torch.from_numpy(self.X[1][index]).float()
            if self.transforms is not None:
                X_unsupervised_tensor = self.transforms(X_unsupervised_tensor)
                X_embedding_tensor = self.transforms(X_embedding_tensor)
                X=(X_embedding_tensor,X_unsupervised_tensor)
            else:
                X=(X_embedding_tensor,X_unsupervised_tensor)
        else:
            X = torch.from_numpy(self.X[index]).float()
            if self.transforms is not None:
                X = self.transforms(X)
        if self.labels is not None:
            return (X, self.labels[index])
        else:
            return (X)
    def __len__(self):
        if isinstance(self.X,tuple):
            return self.X[0].shape[0]
        else:
            return self.X.shape[0]


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)
class DNN(nn.Module):
    '''
    rc: l2 regularization parameter
    input_shape: 2d list or int
        If 2d:
            input_shape[0]: dimension of embedding features for deep network
            input_shape[1]: dimension of unsupervised features for wide network
        If int:
            dimension of embedding features for deep network

    '''
    def __init__(self,
                 hiddens=[500],
                 dropout=0.1,
                 tol=1e-4,
                 rc=1e-3,
                 batch_size=16,
                 lr=1e-3,
                 num_epochs=1000,
                 patience=20,
                 verbose=False,
                 **para):
        super(DNN, self).__init__()
        # the initilization only sets up necessary parameters.
        # the layers information is initialized when `.fit()` function is called.
        use_cuda = torch.cuda.is_available()
        self.verbose=verbose
        if self.verbose:
            print('use_cuda: ' + str(use_cuda))
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.kwargs = {'num_workers': 1, 'pin_memory': False} if use_cuda else {}
        self.para= {'tol':tol,
                    'rc': rc,
                    'batch_size':batch_size,
                    'lr':lr,
                    'num_epochs':num_epochs,
                    'patience':patience,
                    'hiddens':hiddens,
                    'dropout':dropout,
                    }
        self.para.update(para)

    def initiate_model(self):
        input_shape=self.input_shape
        if isinstance(input_shape,tuple):
            embed_shape=input_shape[0]
        else:
            embed_shape=input_shape
        # para['hiddens'] is the ratio of hidden layer size comparing to the input size
        # get the size of hidden layers from these ratios, stored in 'hiddens'
        hiddens=[]
        for h in self.para['hiddens']:
            hiddens.append(math.ceil(h*embed_shape))
        dropout=self.para['dropout']
        if isinstance(input_shape,tuple):
            assert len(input_shape)==2; 'input_shape must be a 2d list or an integer'
            hiddens.insert(0, input_shape[0])
            self.norm = nn.BatchNorm1d(hiddens[-1] + input_shape[1], eps=10 ** (-5))
            self.out = nn.Linear(hiddens[-1] + input_shape[1], self.D_out)
        else:
            assert isinstance(input_shape,int); 'input_shape must be a 1d list or an integer'
            hiddens.insert(0,input_shape)
            self.out = nn.Linear(hiddens[-1], self.D_out)
        # input layer and initialize weights
        # self.BatchNormalization = nn.ModuleList([nn.BatchNorm1d(H[i],eps=10**(-5)) for i in range(len(H)-1)])
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(hiddens[i], hiddens[i + 1]),
                self.activation,
                nn.BatchNorm1d(hiddens[i+1],eps=10**(-5)),
            )
                for i in range(len(hiddens) - 1)
            ]
        )
        self.to(self.device)
    def forward(self, X):


        if isinstance(X,tuple):
            # X=(X1,X2): deep-wide neural network
            assert len(X)==2;"Tuple type features must have two elements"
            X_embedding,X_unsupervised=X

            for layer in self.fc:
                X_embedding = self.dropout(layer(X_embedding))
            if X_unsupervised.shape[1]>0 and X_embedding.shape[1]>0:
                # if "reg_coef" in self.para:
                #     X = torch.cat((X_embedding, X_unsupervised/sqrt(self.para["reg_coef"])), dim=1)
                # else:
                X = torch.cat((X_embedding, X_unsupervised), dim=1)
                X=self.norm(X)
                y =self.out(X)
                # y = self.out(X_embedding)+self.out_augmented(X_unsupervised)
            elif X_embedding.shape[1]>0:
                y = self.out(X_embedding)
            elif X_unsupervised.shape[1]>0:
                y = self.out(X_unsupervised)
            else:
                y=0
        else:
            # only one feature X: neural network
            for layer in self.fc:
                X = self.dropout(layer(X))
            y = self.out(X)
        return y

    def test(self, test_loader):
        self.eval()  # tell that you are testing, == model.train(model=False)
        device=self.device
        test_loss = 0.0
        TARGET = np.zeros(0)
        OUTPUT = np.zeros(0)
        with torch.no_grad():
            for X, target in test_loader:
                X=device_tuple(X,device)
                target = target.to(device).float()
                output = self(X).view(-1, 1)
                target = target.view(-1, 1)
                test_loss += F.mse_loss(output, target, reduction='sum').item()
                TARGET = np.append(TARGET, target.cpu().numpy())
                OUTPUT = np.append(OUTPUT, output.cpu().numpy())
            # test_loss /= len(test_loader.dataset)
        # test_loss = sqrt(test_loss)
        test_loss = sqrt(test_loss / len(test_loader.dataset))
        return OUTPUT, TARGET, test_loss

    def run_epoch(self, train_loader, criterion, optimizer):
        device = self.device
        loss_train = 0.0
        for (X, target) in train_loader:
            if len(target)>1:
                self.train()  # tells your model that you are training the model
                X = device_tuple(X, device)
                target = target.to(device).float()
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                output = self(X).view(-1, 1)
                target = target.view(-1, 1)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                loss_train += loss.item()
        loss_train = sqrt(loss_train / len(train_loader))
        return loss_train

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        # initiate the model by taking the shape of 'X_train' as an input

        self.input_shape=input_shape_tuple(X_train)

        self.D_out=1
        self.initiate_model()
        kwargs = self.kwargs
        para = self.para
        device=self.device
        best_l2=100000.0
        p=0
        e=0
        patience=para['patience']
        tol=para['tol']
        train_dataset = Augmented_Dataset(X_train, y_train)

        train_loader = DataLoader(dataset=train_dataset, batch_size=para['batch_size'], shuffle=True, **kwargs)
        # if y_val is not None:
        #     val_dataset = Augmented_Dataset(X1_val, X2_val, y_val)
        #     val_loader = DataLoader(dataset=val_dataset, batch_size=para['batch_size'], shuffle=False, **kwargs)
        # model = DNN([X1_train.shape[1], X2_train.shape[1]],para['dropout'],para['hiddens']).to(device)
        self.apply(initialize_weights)
        # if X2_train.shape[1]>0:
        #     self.out_augmented.weight.data.fill_(1.0)
        criterion = nn.MSELoss()
        optim_params=[
            {'params': self.parameters(), 'lr': para['lr'], 'weight_decay': para['rc']},
            # {'params': self.out.parameters(), 'lr': para['lr'], 'weight_decay': para['rc2']},
        ]
        optimizer = optim.Adam(optim_params)
        # lr_adjust = optim.lr_scheduler.MultiStepLR(optimizer,
        #                                       milestones=para['milestones'],
        #                                       gamma=para['gamma'],
        #                                       last_epoch=-1)

        # optimizer = optim.SGD(optim_params,lr=para['lr'])
        for epoch in range(para['num_epochs']):
            ltrain = self.run_epoch(train_loader, criterion, optimizer)
            # lr_adjust.step()
            if np.mod(epoch,10)==0 and self.verbose:
                print('epoch=' + str(epoch)+' loss='+str(ltrain))
            # if y_val is not None:
            #     y_pred, y_real, l2_val = self.test(val_loader)

            if best_l2>ltrain+tol:
                # best_l2=ltrain
                e=epoch
                p=0
            else:
                p+=1
            if best_l2>ltrain:
                best_l2 = ltrain
            if p >= patience:
                if self.verbose:
                    print('total epochs='+str(e))
                    print('traning loss='+str(ltrain))
                return e
        print('warning: Max number of epochs is reached. Model may not be accurate.')
        return e

    def predict(self, X_test):
        kwargs=self.kwargs
        test_dataset = Augmented_Dataset(X_test)
        test_loader = DataLoader(dataset=test_dataset, batch_size=self.para['batch_size'], shuffle=False, **kwargs)

        self.eval()  # tell that you are testing, == model.train(model=False)
        device=self.device
        test_loss = 0.0
        OUTPUT = np.zeros(0)
        with torch.no_grad():
            for X in test_loader:
                X = device_tuple(X, device)
                output = self(X).view(-1, 1)
                OUTPUT = np.append(OUTPUT, output.cpu().numpy())
        return OUTPUT
