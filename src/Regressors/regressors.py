import numpy as np
import pandas as pd
import os,sys,warnings

from sklearn.model_selection import KFold

from tqdm import tqdm
from math import sqrt

# import custom modules
from .defaults import DEFAULT_TRAINING_PARAMS, DEFAULT_MODEL_PARAMS
from .mlde_classes import MldeModel
from .mlde_hyperopt import run_hyperopt
# from train_and_predict import train_and_predict
# from utils import mse
from utils import mse,spearman


def prep_input_data(parameter_df):
    """
    Given the data input to run_mlde_cl, build all args needed for running both
    default and hyperparameter optimization functions. This means instantiating
    a number of model instances with inbuilt default parameters (for passage
    into run_mlde) as well as packaging args needed for run_hyperopt_mlde()

    Parameters
    ----------
    parameter_df: pd.DataFrame
        Dataframe derived from MLDE.Support.Params.MldeParameters.csv, containing
        only those models for which Include is True.

    Returns
    -------
    mods_for_default: Iterable of MldeModel instances
        MldeModel instances to be passed into run_mlde()
    hyperopt_args: Iterable of tuples
        Arguments to pass into run_hyperopt_mlde()
    model_params: List of dict
        Default model parameters for models in the order of parameter_df
    """
    # Make sure the shape is 3D
    # assert len(x_shape) == 3, "Input shape should be 3D"

    # Create an empty list in which to store objects
    n_mods = len(parameter_df)
    mods_for_default = [None for _ in range(n_mods)]
    hyperopt_args = [None for _ in range(n_mods)]
    model_params = [None for _ in range(n_mods)]
    for i, (_, row) in enumerate(parameter_df.iterrows()):

        # Pull info needed to instantiate model
        major_model = row["ModelClass"]
        specific_model = row["SpecificModel"]

        # Define the model and training parameters
        model_params[i] = DEFAULT_MODEL_PARAMS[major_model][specific_model].copy()

        # Instantiate a model with default parameters
        mods_for_default[i] = MldeModel(major_model, specific_model,
                                        model_params=model_params[i],
                                        training_params=DEFAULT_TRAINING_PARAMS[major_model],
                                        eval_metric=mse)

        # Package args for hyperopt
        hyperopt_args[i] = (major_model, specific_model, row["NHyperopt"])

    # Return the instantiated models and the hyperopt args
    return mods_for_default, hyperopt_args,model_params

def run_mlde(mlde_models, sampled_x, sampled_y,train_test_inds=None,progress_pos=0,_seeds=None):
    """
    Runs MLDE starting from a list of gpu- and cpu-bound MldeModel instances
    and reported training data. Designed as a method to enable users to perform
    MLDE using a custom model.

    Parameters
    ----------
    mlde_models: iterable of MldeModel instances
        An iterable of pre-instantiated MldeModels.
        These models will be run in series.
    sampled_x: Tuple with two 2D numpy array or One 2D numpy array
        Features.
        Tuple: x[0] features from protein embedding
               x[1] features from unsupervised model
    sampled_y: 1D numpy array
        Numpy array containing the labels (fitness) for training combinations
    train_test_inds: list of lists:
        Cross-validation indices for the run
    progress_pos: int
        Passed to tqdm. Gives the location of the process bar.

    Returns
    -------
    training_loss: List of float
        Mean of cross-validation training error
        for the models in the order they were passed in run_mlde() and run_hyperopt()
    testing_loss: List of float
        Mean of cross-validation testing error
        for the models in the order they were passed in run_mlde() and run_hyperopt()
    model_names: List of str
        Model names
    """
    # Confirm we have the correct number of seeds
    if _seeds is not None:
        assert len(_seeds) == len(mlde_models), "Expect equal numbers of seeds and models"

    # If progress pos is None, disable tqdm
    if progress_pos is None:
        disable = True
        progress_pos = 0
    else:
        disable = False

    # Get the model names
    model_names = np.array([f"{model.major_model};{model.specific_model}"
                            for model in mlde_models])

    # Package args for both GPU and CPU
    model_args = [[model, train_test_inds] for model in mlde_models]

    # Now run all models
    n_models = len(model_names)
    training_loss = [None for _ in range(n_models)]
    testing_loss = [None for _ in range(n_models)]

    for i, (model, train_test_inds) in tqdm(enumerate(model_args),
                                            desc="Default Training",
                                            position=progress_pos, total=n_models,
                                            leave=False, disable=disable):
        print(model.specific_model)
        # Seed if requested
        if _seeds is not None:
            seed_int = _seeds[i]
            # Seed sklearn. Sklearn uses numpy random throughout.
            np.random.seed(seed_int)
            torch.manual_seed(seed_int)
            # Note that XGBoost is deterministic as used in this package - no
            # seeding required
        # Run MLDE
        # results[i] = train_and_predict(model,
        #                                sampled_x=sampled_x,
        #                                sampled_y=sampled_y,
        #                                x_to_predict=x_to_predict,
        #                                train_test_inds=train_test_inds,
        #                                )
        training_loss[i], testing_loss[i] = model.train_cv(sampled_x, sampled_y,
                                                     train_test_inds)
    return training_loss, testing_loss, model_names

def run_hyperopt_mlde(model_data, sampled_x, sampled_y,train_test_inds, progress_pos=1):
    """
    Function for performing hyperparameter optimization on inbuilt MLDE models.

    Parameters
    ----------
    model_data: iterable
        The input data for models that are not
    sampled_x: 2D numpy array
        Numpy array containing the embeddings for the training data sequences
    sampled_y: 1D numpy array
        Numpy array containing the labels (fitness) for training combinations

    train_test_inds: list of lists:
        Cross-validation indices for the run
    progress_pos: int
        Passed to tqdm. Gives the location of the process bar.

    Returns
    -------
    all_trial_info: pd.DataFrame
        The results of "process_trials" post hyperparameter optimization
    training_loss: List of float
        Mean of cross-validation training error
        for the models in the order they were passed in run_mlde() and run_hyperopt()
    testing_loss: List of float
        Mean of cross-validation testing error
        for the models in the order they were passed in run_mlde() and run_hyperopt()
    best_params: List of dict
        Best parameters selected from hyperopt with lowest cross-validation testing loss
        for the models in the order they were passed in run_mlde() and run_hyperopt()
    """
    # Run hyperopt in series for GPU-based models
    n_models = len(model_data)
    all_trial_info = [None for _ in range(n_models)]
    training_loss = [None for _ in range(n_models)]
    testing_loss = [None for _ in range(n_models)]
    best_params = [None for _ in range(n_models)]
    for i, (major_model, specific_model, hyperopt_rounds) in tqdm(enumerate(model_data),
                                                                  desc="Hyperopt",
                                                                  position=progress_pos,
                                                                  total=n_models,
                                                                  leave=False):
        # If we do not have >0 hyperopt rounds, return filler info
        if hyperopt_rounds <= 0:
            # Define the dud dataframe
            # columns = ["MajorModel", "SpecificModel", "HyperRound", "RunTime",
            #            "TrainErr", "TestErr", "Hyper", "HyperVal"]
            # dud_df = pd.DataFrame([[major_model, specific_model, hyperopt_rounds,
            #                         0, np.inf, np.inf, np.nan, "NoHyperoptPerformed"]],
            #                       columns=columns)
            #
            # # Record the dud results from train and predict
            # n_to_predict = len(x_to_predict)
            # dud_tp = (np.inf, np.inf, np.zeros(n_to_predict), np.zeros(n_to_predict))
            # results[i] = (dud_df, dud_tp)
            testing_loss[i]=10**10
            training_loss[i]=10**10
            continue

        # Pull the default training params for the model
        training_params = DEFAULT_TRAINING_PARAMS[major_model].copy()

        # Run hyperparameter optimization
        all_trial_info[i],training_loss[i], testing_loss[i],best_params[i] = run_hyperopt(major_model, specific_model,
                                                         training_params,
                                                         sampled_x=sampled_x,
                                                         sampled_y=sampled_y,
                                                         eval_metric=mse,
                                                         train_test_inds=train_test_inds,
                                                         hyperopt_rounds=hyperopt_rounds)

    return all_trial_info,training_loss,testing_loss,best_params

def combine_results(default_training_loss,default_testing_loss,default_params, opt_training_loss,opt_testing_loss,hyperopt_params):
    """
    Combines the results from the default model training and hyperparameter
    optimization, using the results from the best model identified between the
    two processes.

    Parameters
    ----------
    default_training_loss: 1d numpy array of float
        the training loss from cross validation using default hyperparameters
        for models in the order they were passed in run_mlde() and run_hyperopt()
    default_testing_loss: 1d numpy array of float
        the testing loss from cross validation using default hyperparameters
        for models in the order they were passed in run_mlde() and run_hyperopt()
    default_params: List
        The default model parameters in the order they were passed in run_mlde() and run_hyperopt()
    opt_training_loss: 1d numpy array of float
        the optimal training loss from cross validation selected from hyperopt
        for models in the order they were passed in run_mlde() and run_hyperopt()
    opt_testing_loss: 1d numpy array of float
        the optimal testing loss from cross validation selected from hyperopt
        for models in the order they were passed in run_mlde() and run_hyperopt()
    hyperopt_params: List
        The optimal model parameters for models in the order they were passed in run_mlde() and run_hyperopt()


    Returns
    -------
    compiled_results: Tuple
        Results formatted as if they were produced by
        MLDE.Support.RunMlde.TrainAndPredict.train_and_predict(). The results for
        the model with the lowest test error between default hyperparameters and
        hyperopt is returned.
    """
    # Unpack hyperopt results
    # trial_info, hyperopt_training_loss,hyperopt_testing_loss,hyperopt_params = hyperopt_results
    # Concatenate and save trial info
    # concatenated_trials = pd.concat(trial_info)

    # if not _debug:
    #     concatenated_trials.to_csv(os.path.join(output, "HyperoptInfo.csv"),
    #                                index=False)

    # Create lists in which we will store combined data
    n_results = len(default_testing_loss)
    assert len(default_testing_loss) == len(opt_testing_loss)
    compiled_testing_loss = [None for _ in range(n_results)]
    compiled_training_loss = [None for _ in range(n_results)]
    best_params = [None for _ in range(n_results)]

    # Loop over the default and hyperopt results
    for i, (default_test_err, opt_test_err) in enumerate(zip(default_testing_loss,opt_testing_loss)):

        # Identify test errors
        # default_test_err = default_result[1]
        # opt_test_err = hyperopt_result[1]

        # Record the results of whichever has lower error
        if default_test_err <= opt_test_err:
            compiled_testing_loss[i] = default_test_err
            compiled_training_loss[i] = default_training_loss[i]
            best_params[i]=default_params[i]
        else:
            compiled_testing_loss[i] = opt_test_err
            compiled_training_loss[i] = opt_training_loss[i]
            best_params[i]=hyperopt_params[i]
    # Return the compiled results
    return compiled_training_loss,compiled_testing_loss,best_params

def process_results(training_loss,testing_loss, model_names, params,n_to_average):
    """
    Select top n_to_averages models according to the testing loss.
    Return both names and parameters for top models

    Parameters
    ----------
    testing_loss: 1d numpy array of float
        Testing loss from cross validation of the models in the order they were passed in run_mlde() and run_hyperopt()
    training_loss: 1d numpy array of float
        Training loss from cross validation of the models in the order they were passed in run_mlde() and run_hyperopt()
    model_names: 1d numpy array of str
        The names of the models in the order they were passed in run_mlde() and run_hyperopt()
    params: List of dict
        Optimal hyperparameters for the models in the order they were passed in run_mlde() and run_hyperopt()
        In each model, the hyperparameters were selected from the default and the hyperopt values.
    n_to_average: int
        The number of top models whose predictions should be averaged together
        to generate final predictions

    Returns
    -------
    top_models_name: 1d numpy array of str
        The names of the top n_to_average models with loweset testing loss from cross validation
    top_models_params: List of dict
        The hyperparameters for the n_to_average top models with loweset testing loss from cross validation
    sorted_testing_loss: 1d numpy array
        The testing loss for top models
    sorted_training_loss:1d numpy array
        The training loss for top models
    """
    # Unpack the unprocessed results
    # all_train_loss, all_test_loss = unprocessed_results

    # Convert to numpy arrays
    # all_train_loss = np.array(all_train_loss)
    all_test_loss = np.array(testing_loss)
    # all_preds = np.stack(all_preds)
    # all_stds = np.stack(all_stds)

    # Get order models from best to worst test loss
    sorted_inds = np.argsort(testing_loss)

    # Sort all other arrays accordingly
    # sorted_train_loss = all_train_loss[sorted_inds]
    # sorted_test_loss = all_test_loss[sorted_inds]
    # sorted_preds = all_preds[sorted_inds]
    # sorted_stds = all_stds[sorted_inds]
    sorted_model_names = model_names[sorted_inds]
    sorted_model_params=[params[i] for i in sorted_inds]
    # Build a dataframe which summarizes results
    # summary_list = [[model, train_loss, test_loss] for model, train_loss, test_loss in
    #                 zip(sorted_model_names, sorted_train_loss, sorted_test_loss)]
    # summary_df = pd.DataFrame(summary_list,
    #                           columns=["ModelName", "cvTrainingError", "cvTestingError"])

    # Generate compound predictions
    # cumulative_preds = np.cumsum(sorted_preds, axis=0)
    # compound_preds = np.empty_like(sorted_preds)
    # for i in range(len(compound_preds)):
    #     compound_preds[i] = cumulative_preds[i] / (i + 1)
    #
    # # Pull the requested averaged value
    # mlde_preds = compound_preds[n_to_average - 1]
    top_models_name=sorted_model_names[0:n_to_average]
    top_models_params=sorted_model_params[0:n_to_average]
    sorted_training_loss=training_loss[sorted_inds[0:n_to_average]]
    sorted_testing_loss=testing_loss[sorted_inds[0:n_to_average]]
    return top_models_name,top_models_params,sorted_training_loss,sorted_testing_loss
def ensemble_predictions(sampled_x,sampled_y,x_to_predict,top_models_name,top_models_params,n_to_average):
    """
    Train top models selected from cross validation.
    Optimal models parameters are used and they are trained on all training data.

    Parameters
    ----------
    sampled_x: 2D numpy array
        Numpy array containing the embeddings for the training data sequences
    sampled_y: 1D numpy array
        Numpy array containing the labels (fitness) for training combinations
    x_to_predict: 2D Numpy array
        Numpy array containing the embeddings for the testing data sequences
    top_models_name: 1d numpy array of str
        The names of the top n_to_average models with loweset testing loss from cross validation
    top_models_params: List of dict
        The hyperparameters for the n_to_average top models with loweset testing loss from cross validation
    n_to_average: int
        The number of top models whose predictions should be averaged together
        to generate final predictions

    Returns
    -------
    preds: 1d numpy array
        Ensemble fitness predictions from top models for mutations with features `x_to_predict`
    """
    preds=[None for _ in range(n_to_average)]
    for i in tqdm(range(n_to_average),desc="Ensemble models training",
                  position=0,total=n_to_average,
                  leave=False, disable=False):
        major_model,specific_model=top_models_name[i].split(';')
        model = MldeModel(major_model, specific_model,
                                        model_params=top_models_params[i],
                                        training_params=DEFAULT_TRAINING_PARAMS[major_model],
                                        eval_metric=mse)
        model.train(sampled_x,sampled_y)
        preds[i],_= model.predict(x_to_predict)
    preds=np.array(preds)
    # preds=np.mean(preds,axis=0)
    return preds

def EnsembleRegressors(X_train,Y_train,X_test,reg_para,hyperopt = True,
                       n_to_average = 3,n_cv = 5,_seeds = None):
    reg_para_csv=pd.read_csv(reg_para)
    limited_df = reg_para_csv.loc[reg_para_csv.Include, :]
    kfold_splitter = KFold(n_splits=n_cv)
    train_test_inds = list(kfold_splitter.split(Y_train))
    n_models = len(limited_df)

    run_CV=True
    if n_to_average > n_models:
        warnings.warn(f"Requested averaging {n_to_average}, but only {n_models} will be trained. Averaging all models.")
        n_to_average = n_models

        if not hyperopt: # using all models without hyperopt, then model selection is not performed
            run_CV=False

    if run_CV:
        model_info = limited_df.copy()
        default_mods, mod_hyperopts, default_params = prep_input_data(model_info)
        default_training_loss, default_testing_loss, model_names = run_mlde(default_mods,
                                                                            sampled_x=X_train,
                                                                            sampled_y=Y_train,
                                                                            train_test_inds=train_test_inds,
                                                                            progress_pos=0,
                                                                            )
        if hyperopt:
            opt_trial_info, opt_training_loss, opt_testing_loss, hyperopt_params \
                = run_hyperopt_mlde(mod_hyperopts, X_train, Y_train,
                                    train_test_inds, progress_pos=0)

            training_loss, testing_loss, best_params = combine_results(default_training_loss,
                                                                       default_testing_loss,
                                                                       default_params,
                                                                       opt_training_loss,
                                                                       opt_testing_loss,
                                                                       hyperopt_params
                                                                       )

        else:
            best_params = default_params
            testing_loss = default_testing_loss
            training_loss = default_training_loss
        testing_loss_cv = np.array(testing_loss)
        training_loss_cv = np.array(training_loss)


        # print(len(best_params))
        top_models_name, top_models_params,sorted_training_loss,sorted_testing_loss \
            = process_results(training_loss_cv, testing_loss_cv, model_names, best_params, n_to_average)
        # print(hyperopt_params)
    else:
        model_info = limited_df.copy()
        default_mods, mod_hyperopts, default_params = prep_input_data(model_info)
        model_names = np.array([f"{model.major_model};{model.specific_model}"
                                for model in default_mods])
        top_models_name=model_names
        top_models_params=default_params
        testing_loss_cv=np.inf*np.zeros(n_to_average)
        training_loss_cv=np.inf*np.zeros(n_to_average)
    preds = ensemble_predictions(X_train, Y_train, X_test, top_models_name,
                                 top_models_params,n_to_average)
    # print(top_models_name)
    # print(top_models_params)
    # for i in range(n_to_average):
    #     print('spearman and rmse' + 'model number ' + str(i))
    #     print(spearman(mlde_preds[i, :], Y_test))
    #     print(sqrt(mse(mlde_preds[i, :], Y_test)))

    # print('ensemble results')
    # cumulative_preds = np.cumsum(mlde_preds, axis=0)
    # for i in range(len(cumulative_preds)):
    #     cumulative_preds[i, :] = cumulative_preds[i, :] / (i + 1)
    #     print(spearman(cumulative_preds[i, :], Y_test))

    # print(spearman(np.mean(mlde_preds,axis=0),Y_test))
    # print(top_models)


    return preds,top_models_name,top_models_params,training_loss_cv,testing_loss_cv