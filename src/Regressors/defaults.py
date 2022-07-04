"""
This file contains all default training and model parameters for MLDE. It also
tells the program which models are designed to be run on the GPU and which are 
designed to be run on the CPU. Changing default parameters in here will change
the defaults used. Changing which models are assigned as CPU and GPU will just
break the program. 

Note that all Keras parameters are defined relative to the input shape of the
data (e.g. size parameters are floats between 0 and 1). 
"""
# Define the default model parameters
rc=1e-2
dropout=0.5
size1=0.1
size2=0.05
size3=0.025
DEFAULT_MODEL_PARAMS = {"DNN": {"OneHidden": {"rc":rc,
                                              "dropout": dropout,
                                              },
                                 "TwoHidden": {"rc":rc,
                                               "dropout": dropout,
                                               },
                                "ThreeHidden":{"rc":rc,
                                               "dropout": dropout,
                                               },
                                 },
                        "DNN_deep":{
                                    "OneHidden": {"rc": 1e-4,
                                            "dropout": 0.5, },
                                    "TwoHidden": {"rc": 1e-4,
                                            "dropout": 0.5, },
                                    "ThreeHidden":{"rc":1e-4,
                                                  "dropout":0.5,},
                                    "FourHidden": {"rc": 1e-4,
                                                    "dropout": 0.5, },
                                    "FiveHidden":{"rc":1e-4,
                                                  "dropout":0.5,},
                                    },
                      "XGB": {"Tree": {"eta": 0.3,
                                       "max_depth": 6,
                                       "lambda": 1,
                                       "alpha": 0,
                                       "num_boost_round":500,},
                              "Linear": {"lambda": 1,
                                         "alpha": 0,
                                         "num_boost_round":500,},
                              "Tree-Tweedie": {"eta": 0.3,
                                               "max_depth": 6,
                                               "lambda": 1,
                                               "alpha": 0,
                                               "num_boost_round":500,},
                              "Linear-Tweedie": {"lambda": 1,
                                                 "alpha": 0,
                                                 "num_boost_round":500,}
                              },
                      "sklearn-regressor": {"Linear": {},
                                            "GradientBoostingRegressor": {"n_estimators": 500},
                                            "RandomForestRegressor": {"n_estimators": 500},
                                            "BayesianRidge": {},
                                            "LinearSVR": {},
                                            "ARDRegression": {},
                                            "KernelRidge": {'kernel':'rbf'},
                                            "BaggingRegressor": {"n_estimators": 500},
                                            "LassoLarsCV": {"cv": 5},
                                            "DecisionTreeRegressor": {},
                                            "SGDRegressor": {},
                                            "KNeighborsRegressor": {},
                                            "ElasticNet": {},
                                            "Ridge":{}
                                            }
                      }

# Define the default training parameters
DEFAULT_TRAINING_PARAMS = { "DNN_deep":{"patience": 20,
                                   "batch_size": 64,
                                   "tol":1e-4,
                                   "num_epochs": 2000,
                                   "lr":1e-3,
                                   "size1": 1.0,
                                   "size2": 1.0,
                                   "size3": 1.0,
                                   "size4": 1.0,
                                   "size5": 1.0,
                                   # "gamma":0.1,
                                   # "milestones":[50],
                                   'verbose':False
                            },
                            "DNN": {"patience": 20,
                                   "batch_size": 64,
                                   "tol":1e-4,
                                   "num_epochs": 1000,
                                   "lr":1e-3,
                                   "size1": size1,
                                   "size2": size2,
                                   "size3": size3,
                                   # "gamma":0.1,
                                   # "milestones":[50],
                                   'verbose':False},
                         "XGB": {#"early_stopping_rounds": 10,
                                 # "num_boost_round": 1000,
                                 "verbose_eval": False},
                         "sklearn-regressor": {}
                         }

# Define the CPU models
CPU_MODELS = (("XGB", "Tree"),
              ("XGB", "Linear"),
              ("XGB", "Tree-Tweedie"),
              ("XGB", "Linear-Tweedie"),
              ("sklearn-regressor", "Linear"),
              ("sklearn-regressor", "GradientBoostingRegressor"),
              ("sklearn-regressor", "RandomForestRegressor"),
              ("sklearn-regressor", "BayesianRidge"),
              ("sklearn-regressor", "LinearSVR"),
              ("sklearn-regressor", "ARDRegression"),
              ("sklearn-regressor", "KernelRidge"),
              ("sklearn-regressor", "BaggingRegressor"),
              ("sklearn-regressor", "LassoLarsCV"),
              ("sklearn-regressor", "DecisionTreeRegressor"),
              ("sklearn-regressor", "SGDRegressor"),
              ("sklearn-regressor", "KNeighborsRegressor"),
              ("sklearn-regressor", "ElasticNet"),
              ("sklearn-regressor", "Ridge")
              )

# Define the GPU models
GPU_MODELS = (("DNN", "OneHidden"),
              ("DNN", "TwoHidden"),
              ("DNN", "ThreeHidden"),
              ("DNN_deep", "OneHidden"),
              ("DNN_deep", "TwoHidden"),
              ("DNN_deep", "ThreeHidden"),
              ("DNN_deep", "FourHidden"),
              ("DNN_deep","FiveHidden")
              )