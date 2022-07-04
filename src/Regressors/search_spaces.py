"""
This file contains all information relevant to running hyperopt with MLDE. The 
only things that can be safely changed in this file are the ranges and priors
on the search spaces defined in 'search_spaces'
"""
# Import relevant modules
from hyperopt import hp

# Define generic parameters that will be searched over
_deep_space=("dropout","rc")


_nohidden_space = ("dropout",)
_onehidden_space = ("dropout","rc")
_twohidden_space = ("dropout","rc")
_threehidden_space=("dropout","rc")

_xgbtree_space = ("eta", "max_depth", "lambda", "alpha","num_boost_round")
_xgblinear_space = ("lambda", "alpha","num_boost_round")

_linear_space = ("dummy",)
_gradientboosting_space = ("learning_rate", "n_estimators_gbt", "min_samples_split",
                          "min_samples_leaf", "max_depth")
_randomforest_space = ("n_estimators", "min_samples_split", "min_samples_leaf",
                      "max_depth")
_linearsvr_space = ("tol", "C", "dual")
_ardregression_space = ("tol", "alpha_1", "alpha_2", "lambda_1", "lambda_2")
_kernelridge_space = ("alpha", "kernel")
_bayesianridge_space = ("tol", "alpha_1", "alpha_2", "lambda_1", "lambda_2")
_bagging_space = ("n_estimators", "max_samples")
_lassolars_space = ("max_iter", "cv", "max_n_alphas")
_decisiontreeregressor_space = ("max_depth", "min_samples_split",
                               "min_samples_leaf")
_sgdregressor_space = ("alpha", "l1_ratio", "tol")
_kneighborsregressor_space = ("n_neighbors", "weights", "leaf_size", "p")
_elasticnet_space = ("l1_ratio", "alpha")
_adaboost_space = ("n_estimators", "learning_rate")
_ridge_space=("alpha",)
# Define choice tuples for each time a choice is made in hyperopt
CATEGORICAL_PARAMS = {"dual": (True, False),
                      "weights": ("uniform", "distance"),
                      "kernel": ("rbf", "laplacian", "polynomial", "chi2", "sigmoid")}

# Define a list of values that must be converted to integers after coming from
# hyperopt
INTEGER_PARAMS = {"min_samples_split","min_samples_leaf","max_depth","num_boost_round", "n_estimators",
                  "n_neighbors", "leaf_size", "max_iter", "cv", "max_n_alphas"}

# Define a set of values that are written in the search space as the fraction of
# total latent space
# LATENT_PERC_PARAMS = {"size1", "size2", "n_filters1", "n_filters2"}
log10=2.30258509299
# Define search spaces for all available parameters
SEARCH_SPACES = {"DNN_deep":{"dropout": hp.uniform("dropout", 0.2, 0.8),
                  "size1": hp.uniform("size1", 0.5, 2.),
                  "size2": hp.uniform("size2", 0.5, 2.),
                  "size3": hp.uniform("size2", 0.5, 2.),
                  "size4": hp.uniform("size2", 0.5, 2.),
                  "size5": hp.uniform("size2", 0.5, 2.),
                  "rc": hp.loguniform("rc",-5*log10,0*log10),
                             },
            "XGB": {"eta": hp.loguniform("eta", -2*log10, 0*log10),
                  "max_depth": hp.quniform("max_depth", 3, 8, 1),
                  "lambda": hp.loguniform("lambda", -3*log10, 2**log10),
                  "alpha": hp.loguniform("alpha", -3*log10, 2**log10),
                 "num_boost_round":hp.quniform("num_boost_round",100, 2000,10)},
          "DNN": {"dropout": hp.uniform("dropout", 0.2, 0.95),
                  "size1": hp.uniform("size1", 0.25, 0.75),
                  "size2": hp.uniform("size2", 0.25, 0.75),
                  "size3": hp.uniform("size2", 0.25, 0.75),
                  "rc": hp.loguniform("rc",-4*log10,0*log10),
                  },
          "sklearn-regressor": {"n_estimators_gbt": hp.quniform("n_estimators", 100, 2000, 10),
                      "n_estimators": hp.quniform("n_estimators", 100, 2000, 10),
                      "learning_rate": hp.loguniform("learning_rate", -3*log10, 0*log10),
                      "max_samples": hp.uniform("max_samples", 0.1, 1),
                      "max_depth": hp.quniform("max_depth", 3, 8, 1),
                      "min_samples_split": hp.quniform("min_samples_split", 2, 8,1),
                      "min_samples_leaf": hp.quniform("min_samples_leaf", 1, 8,1),
                      # "subsample":hp.uniform("subsample",0.2,0.8),
                      "tol": hp.loguniform("tol", -5*log10, -3*log10),
                      "C": hp.uniform("C", 0.1, 10),
                      "dual": hp.choice("dual", CATEGORICAL_PARAMS["dual"]),
                      "alpha_1": hp.loguniform("alpha_1", -7*log10, -5*log10),
                      "alpha_2": hp.loguniform("alpha_2", -7*log10, -5*log10),
                      "lambda_1": hp.loguniform("lambda_1", -7*log10, -5*log10),
                      "lambda_2": hp.loguniform("lambda_2", -7*log10, -5*log10),
                      "n_neighbors": hp.quniform("n_neighbors", 1, 30, 1),
                      "weights": hp.choice("weights", CATEGORICAL_PARAMS["weights"]),
                      "leaf_size": hp.quniform("leaf_size", 1, 50, 1),
                      "p": hp.uniform("p", 1, 2),
                      "l1_ratio": hp.uniform("l1_ratio", 0, 1),
                      # "alpha": hp.uniform("alpha", 1e-4, 10),
                      "alpha": hp.loguniform("alpha", -4*log10, 5*log10),# (0.01,1000)
                      "kernel": hp.choice("kernel", CATEGORICAL_PARAMS["kernel"]),
                      "max_iter": hp.quniform("max_iter", 10, 1000, 1),
                      "cv": hp.quniform("cv", 2, 10, 1),
                      "max_n_alphas": hp.quniform("max_n_alphas", 10, 2000, 1),
                      "dummy": hp.uniform("dummy", 0, 1)}}

# Define the generic search spaces for each model type
SPACE_BY_MODEL = {"DNN_deep":{"OneHidden":_deep_space,
                              "TwoHidden":_deep_space,
                              "ThreeHidden":_deep_space,
                              "FourHidden":_deep_space,
                              "FiveHidden":_deep_space
                              },
                  "DNN": {"OneHidden": _onehidden_space,
                            "TwoHidden": _twohidden_space,
                          "ThreeHidden": _threehidden_space,
                          },
                  "XGB": {"Tree": _xgbtree_space,
                          "Linear": _xgblinear_space,
                          "Tree-Tweedie": _xgbtree_space,
                          "Linear-Tweedie": _xgblinear_space},
                  "sklearn-regressor": {"Linear": _linear_space,
                                        "GradientBoostingRegressor": _gradientboosting_space,
                                        "RandomForestRegressor": _randomforest_space,
                                        "LinearSVR": _linearsvr_space,
                                        "ARDRegression": _ardregression_space,
                                        "KernelRidge": _kernelridge_space,
                                        "BayesianRidge": _bayesianridge_space,
                                        "BaggingRegressor": _bagging_space,
                                        "LassoLarsCV": _lassolars_space,
                                        "DecisionTreeRegressor": _decisiontreeregressor_space,
                                        "SGDRegressor": _sgdregressor_space,
                                        "KNeighborsRegressor": _kneighborsregressor_space,
                                        "ElasticNet": _elasticnet_space,
                                        "AdaBoostRegressor": _adaboost_space,
                                        "Ridge":_ridge_space,}
                  }
RENAME_VAR={"n_estimators_gbt":"n_estimators"}