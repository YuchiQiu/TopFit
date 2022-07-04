'''
Inner loop in evaluating predictors with multiple processors.
See also evaluate.py.
'''
import functools
import logging
import os
from sklearn.model_selection import KFold

import numpy as np
import pandas as pd

from Encodings import get_encoder_cls, JointEncoder,get_encoder_names
from Regressors import EnsembleRegressors
from Regressors.mlde_utils import single_structure_tuple
from utils.metric_utils import spearman, topk_mean, r2, hit_rate, aucroc, ndcg,rmse
from utils.io_utils import load_data_split, get_wt_log_fitness, get_log_fitness_cutoff
from utils.data_utils import dict2str

import argparse

from utils import parse_vars
MAX_N_TEST=10000


def evaluate_predictor(dataset_name, encoder_name, reg_para,n_structure,
        n_train, max_n_mut, train_n_mut, ignore_gaps,
        seed, encoder_params,n_to_average,hyperopt, outdir
                       ,results_suffix,save_pred,structure_id,cv_id):
    outpath = os.path.join(outdir, f'results{results_suffix}.csv')
    if structure_id is None:
        outpath_npz=os.path.join(outdir, 'results'+results_suffix+'_'
                                 +encoder_name+'_n'+
                                 '_seed'+str(seed)
                                 +'_cv'+str(cv_id)+'.npz')
    else:
        outpath_npz=os.path.join(outdir, 'results'+results_suffix+'_'
                                 +encoder_name+'_n'+
                                 '_seed'+str(seed)+'_strc'+str(structure_id)
                                 +'_cv'+str(cv_id)+'.npz')
    if not os.path.exists(outpath_npz):
        print(f'----- encoder {encoder_name}, seed {seed} -----')
        outpath = f'{outpath}-{os.getpid()}'  # each process writes to its own file
        # data = load_data_split(dataset_name, split_id=-1,
        #         ignore_gaps=ignore_gaps)
        data = pd.read_csv(os.path.join('data', dataset_name, 'data.csv'))
        encoder_cls = get_encoder_cls(encoder_name)
        if len(encoder_cls) == 1 and n_train==0:
            encoder = encoder_cls[0](dataset_name,encoder_name, **encoder_params)
        else:
            encoder = JointEncoder(dataset_name, encoder_cls,
                encoder_name.split('+'), **encoder_params)

        np.random.seed(seed=seed)
        kf = KFold(n_splits=5,random_state=seed,shuffle=True)
        # predictions=np.inf*np.zeros(len(data))

        it=0
        for i, j in kf.split(data):
            if it==cv_id:
                train_index=i
                test_index=j
                train = data.loc[train_index]
                test  = data.loc[test_index]
                break
            else:
                it+=1

        # test = data.sample(frac=0.2, random_state=seed)
        # if len(test) > MAX_N_TEST:
        #     test = test.sample(n=MAX_N_TEST, random_state=seed)
        # test = test.copy()
        # train = data.drop(test.index)
        # if train_n_mut>0 and 'n_mut' in data.columns:
        #     train = train[train.n_mut <= train_n_mut]
        # assert len(train) >= n_train, 'not enough training data'

        regressor_dic = {}
        if n_train == 0:
            """"unsupervised prediction; only one zero-shot predictor involved"""
            test['pred'] = encoder.predict_unsupervised(test.seq.values)
            predictions=test['pred'].values
        else:
            """"
            n_train=-1: 80/20 training/testing split
            """
            # if n_train!=-1:
            #     # downsample to ntrain
            #     train = encoder.select_training_data(train, n_train, seed=seed)
            #     assert len(train) == n_train, (
            #         f'expected {n_train} train examples, received {len(train)}')
            X1_train, X2_train,_ = encoder.seq2feat(train.seq.values)
            Y_train = train.log_fitness.values
            X1_test,X2_test,ensemble_structure=encoder.seq2feat(test.seq.values)

            #single structure data is given
            if not ensemble_structure:
                #consider one single structure
                assert X1_train.shape[1]==X1_test.shape[1];
                "Inconsistent dimension of testing and training data"
                if X2_test.shape[1]==0:
                    X_train=X1_train
                    X_test=X1_test
                else:
                    X_train=(X1_train,X2_train)
                    X_test=(X1_test,X2_test)
                # if regressor=='Ridge':
                #     preds,top_models_name,top_models_params,training_loss_cv,testing_loss_cv=\
                #         RidgeRegressor(X_train,Y_train,X_test)
                # else:
                preds,top_models_name,top_models_params,training_loss_cv,testing_loss_cv=\
                    EnsembleRegressors(X_train,Y_train,X_test,reg_para,hyperopt = hyperopt,
                           n_to_average = n_to_average)
                test['pred'] = np.mean(preds,axis=0)
                predictions = np.mean(preds,axis=0)
                for i,model in enumerate(top_models_name):
                    regressor_dic['top'+str(i)]=model
                    regressor_dic['top'+str(i)+' rmse (train cv)']=training_loss_cv[i]
                    regressor_dic['top'+str(i)+' rmse (test cv)']=testing_loss_cv[i]
                    regressor_dic['top'+str(i)+' para']=dict2str(top_models_params[i])

            # multiple structures are given (e.g. multiple models from NMR data)
            else:
                #consider ensemble of multiple structures.
                assert X1_train.shape[2] == X1_test.shape[2];
                "Inconsistent dimension of testing and training data"
                if n_structure==0:
                    n_structure=X2_test.shape[1]
                if X2_test.shape[2] == 0:
                    X_train = X1_train
                    X_test = X1_test
                else:
                    X_train = (X1_train, X2_train)
                    X_test = (X1_test, X2_test)
                # if regressor=='Ridge':
                #     preds,top_models_name,top_models_params,training_loss_cv,testing_loss_cv=\
                #         RidgeRegressor(X_train,Y_train,X_test)
                # else:

                if structure_id is None:
                    # run regressor on all structures if structure_id is None
                    preds=[None for _ in range(n_structure)]
                    for structure_iter_ID in range(n_structure):
                        x_train = single_structure_tuple(X_train, structure_iter_ID)
                        x_test = single_structure_tuple(X_test, structure_iter_ID)
                        preds[structure_iter_ID], top_models_name, top_models_params, training_loss_cv, testing_loss_cv = \
                            EnsembleRegressors(x_train, Y_train,x_test,
                                               reg_para, hyperopt=hyperopt,
                                               n_to_average=n_to_average)
                    preds=np.asarray(preds)
                    preds=np.mean(preds,axis=0)
                    test['pred'] = np.mean(preds, axis=0)
                    predictions = np.mean(preds, axis=0)
                    regressor_dic['n_structures']=n_structure
                    if n_structure==1:
                        for i, model in enumerate(top_models_name):
                            regressor_dic['top' + str(i)] = model
                            regressor_dic['top' + str(i) + ' rmse (train cv)'] = training_loss_cv[i]
                            regressor_dic['top' + str(i) + ' rmse (test cv)'] = testing_loss_cv[i]
                            regressor_dic['top' + str(i) + ' para'] = dict2str(top_models_params[i])
                else:
                    # run regressor on one structure given by `structure_id`
                    x_train = single_structure_tuple(X_train, structure_id)
                    x_test = single_structure_tuple(X_test, structure_id)

                    preds, _, _, _, _ = \
                        EnsembleRegressors(x_train, Y_train, x_test,
                                           reg_para, hyperopt=hyperopt,
                                           n_to_average=n_to_average)
                    preds=np.asarray(preds)
                    # preds=np.mean(preds,axis=0)
                    test['pred'] = np.mean(preds, axis=0)
                    predictions = np.mean(preds, axis=0)
                    # regressor_dic['n_structures']=n_structure
        # if structure_id is None:
        #     # the csv results are usually saved unless:
        #     # running NMR ensemble model for only one specific strucutre given by `structure_id`
        #     # In this exception, results for each single structure need to be saved in .npz file
        #     # and merge later to generate summary `.csv` file later
        #     summary_csv(dataset_name, encoder_name, n_train, seed, data, test, max_n_mut, regressor_dic, outpath)

        # if save_pred:
        Y=data.log_fitness.values
        np.savez(outpath_npz,dataset_name=dataset_name,
                 encoder_name=encoder_name,
                 n_train=n_train,seed=seed,
                 # data=data,data_key=data.keys(),
                 # test=test,test_key=test.keys(),
                 y_pred=predictions,size=len(data),
                 Y=Y,y_true=test.log_fitness.values,
                 train_index=train_index,test_index=test_index,
                 regressor_dic=regressor_dic,
                 n_structure=n_structure,
                 structure_id=structure_id,
                 results_suffix=results_suffix)

        # return results
def summary_csv(dataset_name,encoder_name,n_train,seed,data,test,max_n_mut,regressor_dic,outpath):
    # save summary results for all metrics
    metric_fns = {
        'spearman': spearman,
        'ndcg': ndcg,
        'rmse': rmse,
        # 'topk_mean': functools.partial(
        #     topk_mean, topk=metric_topk),
        # 'hit_rate_wt': functools.partial(
        #    hit_rate, y_ref=get_wt_log_fitness(dataset_name),
        #    topk=metric_topk),
        # 'hit_rate_bt': functools.partial(
        #    hit_rate, y_ref=train.log_fitness.max(), topk=metric_topk),
        # 'aucroc': functools.partial(
        #    aucroc, y_cutoff=get_log_fitness_cutoff(dataset_name)),
    }
    results_dict = {
        'dataset': dataset_name,
        'encoder': encoder_name,
        'n_train': n_train,
        'seed': seed}
    results_dict.update({k: mf(test.pred.values, test.log_fitness.values)
                         for k, mf in metric_fns.items()})
    if 'n_mut' in data.columns:
        max_n_mut = min(data.n_mut.max(), max_n_mut)
        for j in range(1, max_n_mut + 1):
            y_pred = test[test.n_mut == j].pred.values
            y_true = test[test.n_mut == j].log_fitness.values
            results_dict.update({
                f'{k}_{j}mut': mf(y_pred, y_true)
                for k, mf in metric_fns.items()})

    results_dict.update(regressor_dic)

    results = pd.DataFrame(columns=sorted(results_dict.keys()))
    results = results.append(results_dict, ignore_index=True)
    if os.path.exists(outpath):
        results.to_csv(outpath, mode='a', header=False, index=False,
                       columns=sorted(results.columns.values))
    else:
        results.to_csv(outpath, mode='w', index=False,
                       columns=sorted(results.columns.values))

# def run_from_queue(worker_id, queue):
#     while True:
#         args = queue.get()
#         try:
#             evaluate_predictor(*args)
#         except Exception as e:
#             logging.error("ERROR: %s", str(e))
#             logging.exception(e)
#             queue.task_done()
#         queue.task_done()
def main():
    parser = argparse.ArgumentParser(
            description='Example: python evaluate.py sarkisyan onehot_ridge '
    )
    parser.add_argument('dataset_name', type=str,
            help='Dataset name. Folder of the same name under the data '
            'and inference directories are expected to look up files.'
            'The data will be loaded from data/{dataset_name}/data.csv'
            'in the `seq` and `log_fitness` columns.')
    parser.add_argument('encoder_name', type=str,
            help='Encoder name, or all for running all encoders.')
    # parser.add_argument('--regressor', type=str,
    #         help='perform single regressor if specified; Option: Ridge',default='')
    parser.add_argument('--n_structure', type=int,
            help='Only for running NMR ensemble model.'
                 'number of structures to be considered. '
                 'Default:0 to consider all structures given.',
                        default=0)
    parser.add_argument('--structure_id', type=int,
            help='Only for running NMR ensemble model. '
                 'Default: -1. Run on all structures to generate ensemble predictions in one time.'
                 'If other non-negative values are given, only run on a specific structure. '
                 'Need to ensemble them later. '
                 'NOTE: PLEASE MAKE `save_pred` BEING `TRUE` IF NON-DEFAULT VALUE IS GIVEN',
                 default=-1)
    parser.add_argument('--cv_id', type=int,default=0)
    parser.add_argument('--reg_para', type=str,
                        help='csv documents for hyperopt information',
                        default='Inputs/RegressorPara.csv')
    # parser.add_argument('--n_threads', type=int, default=20)
    parser.add_argument('--n_train', type=int, default=96)
    parser.add_argument('--max_n_mut', type=int, default=5)
    # parser.add_argument('--joint_training', dest='joint_training', action='store_true')
    # parser.add_argument('--boosting', dest='joint_training', action='store_false')
    # parser.set_defaults(joint_training=True)
    # parser.add_argument('--train_on_single', dest='train_on_single', action='store_true')
    # parser.add_argument('--train_on_all', dest='train_on_single', action='store_false')
    # parser.set_defaults(train_on_single=True)
    parser.add_argument('--train_n_mut',type=int,default=0,help='training data selection, number of muation is inlcuded.'
                                                                ' Default is 0 which includes all number of mutations')
    parser.add_argument('--ignore_gaps', dest='ignore_gaps', action='store_true')
    parser.set_defaults(ignore_gaps=False)
    parser.add_argument('--hyperopt_off', dest='hyperopt', action='store_false')
    parser.set_defaults(hyperopt=True)
    parser.add_argument('--n_to_average', type=int, default=3)
    parser.add_argument('--seed', type=int, default=0,
            help='random seed for training and testing data split')
    parser.add_argument('--metric_topk', type=int, default=96,
            help='Top ? when evaluating hit rate and topk mean')
    parser.add_argument("--encoder_params",
                        metavar="KEY=VALUE",
                        nargs='+',
                        help="Set a number of key-value pairs "
                             "(do not put spaces before or after the = sign). "
                             "If a value contains spaces, you should define "
                             "it with double quotes: "
                             'foo="this is a sentence". Note that '
                             "values are always treated as floats.")
    parser.add_argument('--results_suffix', type=str, default='')
    parser.add_argument('--save_pred',help='save predictions to npz. Need to be True is structure_id is not given by default', dest='save_pred', action='store_true')
    parser.set_defaults(save_pred=False)

    args = parser.parse_args()
    encoder_params = parse_vars(args.encoder_params)
    if args.ignore_gaps:
        encoder_params['ignore_gaps'] = args.ignore_gaps
    print(args)

    # outdir = os.path.join('results_cv', args.dataset_name)
    # if not os.path.exists(outdir):
    #     # os.mkdir(outdir)
    #     os.system('mkdir -p '+'results_cv/'+ args.dataset_name)
    # outpath = os.path.join(outdir, f'results{args.results_suffix}.csv')

    outdir = os.path.join('results_cv', args.dataset_name)
    if not os.path.exists(outdir):
        # os.mkdir(outdir)
        os.system('mkdir -p '+'results_cv_'+args.reg_para.split('_')[1][0:-4]+'/'+ args.dataset_name)
    outpath = os.path.join(outdir, f'results{args.results_suffix}.csv')

    encoders = get_encoder_names(args.encoder_name)
    cv_id=args.cv_id
    for pn in encoders:
        if os.path.exists(outpath):
            data = pd.read_csv(outpath)
            data=data[data['dataset']==args.dataset_name]
            data=data[data['n_train']==args.n_train]
            data=data[data['seed']==args.seed]
            data=data[data['encoder']==pn]
            if 'n_structures' in data.columns:
                    # args.structure_id!=-1:
                if args.n_structure==0:
                    data = data[data['n_structures'] >1]
                else:
                    data = data[data['n_structures'] == args.n_structure]
            if len(data)==0:
                run_flag=True
            else:
                run_flag=False
        else:
            run_flag=True
        if run_flag:
            if args.structure_id==-1:
                structure_id=None
            else:
                structure_id=args.structure_id
            evaluate_predictor(args.dataset_name, pn, args.reg_para,
                               args.n_structure, args.n_train, args.max_n_mut,
                               args.train_n_mut, args.ignore_gaps, args.seed,
                               encoder_params, args.n_to_average, args.hyperopt, outdir,args.results_suffix
                               ,args.save_pred,structure_id,cv_id)
if __name__ == '__main__':
    import timeit

    start = timeit.default_timer()

    # Your statements here
    main()
    stop = timeit.default_timer()

    print('Time: ', stop - start)

