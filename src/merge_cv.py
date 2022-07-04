import numpy as np
import pandas as pd
from utils import merge_dfs
import os,sys
from utils.metric_utils import spearman, topk_mean, r2, hit_rate, aucroc, ndcg,rmse

# dataset_name=sys.argv[1]
# results_suffix=sys.argv[2]
def summary_csv(dataset,encoder,seed,Y_pred,Y,outpath):
    # save summary results for all metrics
    metric_fns = {
        'spearman': spearman,
        'ndcg': ndcg,
        'rmse': rmse,
    }
    results_dict = {
        'dataset': dataset,
        'encoder': encoder,
        'n_train': -1,
        'seed': seed}
    results_dict.update({k: mf(Y_pred, Y)
                         for k, mf in metric_fns.items()})
    # if 'n_mut' in data.columns:
    #     max_n_mut = min(data.n_mut.max(), max_n_mut)
    #     for j in range(1, max_n_mut + 1):
    #         y_pred = test[test.n_mut == j].pred.values
    #         y_true = test[test.n_mut == j].log_fitness.values
    #         results_dict.update({
    #             f'{k}_{j}mut': mf(y_pred, y_true)
    #             for k, mf in metric_fns.items()})

    # results_dict.update(regressor_dic)

    results = pd.DataFrame(columns=sorted(results_dict.keys()))
    results = results.append(results_dict, ignore_index=True)
    if os.path.exists(outpath):
        results.to_csv(outpath, mode='a', header=False, index=False,
                       columns=sorted(results.columns.values))
    else:
        results.to_csv(outpath, mode='w', index=False,
                       columns=sorted(results.columns.values))

def run_dataset(dataset_name,regressor, encoder,seed,results_suffix):
    folder='results_cv_'+regressor
    outdir = os.path.join(folder, dataset_name)
    os.system('mkdir '+folder+'/'+dataset_name+'/npz/')
    print(dataset_name+results_suffix)
    file_prefix='results'+results_suffix+'_'+encoder+'_n_seed'+str(seed)+'_cv'
    data=np.load(os.path.join(outdir,file_prefix+str(4)+'.npz'),allow_pickle=True)
    Y=data['Y']
    Y_pred=np.inf*np.zeros(data['size'])
    Y_true=np.inf*np.zeros(data['size'])

    for cv_id in range(5):
        data = np.load(os.path.join(outdir,file_prefix+str(cv_id)+'.npz'), allow_pickle=True)
        Y_pred[data['test_index']]=data['y_pred']
        Y_true[data['test_index']]=data['y_true']
        os.system('mv '+folder+'/'+dataset_name+'/'+file_prefix+str(cv_id)+'.npz '
                  +folder+'/'+dataset_name+'/npz/'+file_prefix+str(cv_id)+'.npz')
    assert np.max(np.abs(Y-Y_true))==0.0
    # print(Y_pred)
    # if not os.path.exists(outdir):
    #     os.mkdir(outdir)
    outpath = os.path.join(outdir, f'results{results_suffix}.csv')
    summary_csv(dataset, encoder, seed, Y_pred, Y_true,outpath)
    # merge_dfs(f'{outpath}*', outpath,
    #           index_cols=['dataset', 'encoder', 'seed', 'n_train'],
    #           groupby_cols=['encoder', 'encoder_params', 'n_train', 'topk'],
    #           ignore_cols=['seed'])
    # if os.path.exists(outpath):
    #     data=pd.read_csv(outpath)
    #     print(len(data))

def run_dataset_NMR(dataset_name,regressor, encoder,seed,results_suffix):
    folder='results_cv_'+regressor
    outdir = os.path.join(folder, dataset_name)
    print(dataset_name+results_suffix)
    file_prefix='results'+results_suffix+'_'+encoder+'_n_seed'+str(seed)+'_strc'
    data=np.load(os.path.join(outdir,file_prefix+str(0)+'_cv0'+'.npz'),allow_pickle=True)
    n_structure=data['n_structure']
    Y=data['Y']
    Y_pred=np.inf*np.zeros(data['size'])
    Y_true=np.inf*np.zeros(data['size'])

    for cv_id in range(5):
        data = np.load(os.path.join(outdir,file_prefix+str(0)+'_cv'+str(cv_id)+'.npz'),
                       allow_pickle=True)
        Y_true[data['test_index']]=data['y_true']
    assert np.max(np.abs(Y-Y_true))==0.0
    for cv_id in range(5):
        y_pred=[]
        for str_id in range(n_structure):
            data = np.load(os.path.join(outdir,
                                        file_prefix + str(str_id) + '_cv' + str(cv_id) + '.npz'),
                           allow_pickle=True)
            y_pred.append(data['y_pred'])
        y_pred=np.array(y_pred)
        y_pred=np.mean(y_pred,axis=0)
        Y_pred[data['test_index']]=y_pred

    # print(Y_pred)
    # if not os.path.exists(outdir):
    #     os.mkdir(outdir)
    outpath = os.path.join(outdir, f'results{results_suffix}.csv')
    summary_csv(dataset, encoder, seed, Y_pred, Y_true,outpath)
    # merge_dfs(f'{outpath}*', outpath,
    #           index_cols=['dataset', 'encoder', 'seed', 'n_train'],
    #           groupby_cols=['encoder', 'encoder_params', 'n_train', 'topk'],
    #           ignore_cols=['seed'])
    # if os.path.exists(outpath):
    #     data=pd.read_csv(outpath)
    #     print(len(data))

if __name__ == "__main__":
    regressor='Ridge'
    for dataset in [
                    # small size datasets:
                    'BLAT_ECOLX_Tenaillon2013-singles-MIC_score',
                    'DLG4_RAT_Ranganathan2012-CRIPT',
                    'GAL4_YEAST_Shendure2015-SEL_C_40h',
                    'RL401_YEAST_Bolon2013-selection_coefficient',
                    'RL401_YEAST_Bolon2014-react_rel',
                    'RL401_YEAST_Mavor2016_DMSO',
                    'MTH3_HAEAESTABILIZED_Tawfik2015-Wrel_G17_filtered',
                    'CALM1_HUMAN_Roth2017_screenscore',
                    'SUMO1_HUMAN_Roth2017_screenscore',
                    'TRPC_THEMA_Chen2017_fitness',
                    'TPMT_HUMAN_Matreyek2019_score',
                    'P84126_THETH_Chen2017_fitness',
                    'YAP1_HUMAN_Fields2012-singles-linear',

                    # medium size dataset
                    'TRPC_SULSO_Chen2017_fitness',
                    'TPK1_HUMAN_Roth2017_screenscore',
                    'RASH_HUMAN_Bandaru2016_unregulated',
                    'UBC9_HUMAN_Roth2017_screenscore',
                    'PTEN_HUMAN_Matreyek2019_score',

                    # NMR datasets:
                    # 'BRCA1_HUMAN_Fields2015-e3_NMR',
                    # 'BRCA1_HUMAN_Fields2015-y2h_NMR',
                    # 'IF1_ECOLI_Kelsic2016_fitness_rich_NMR',
                    # 'UBE4B_MOUSE_Klevit2013-nscor_log2_ratio_single_NMR',

                    # large size datasets:
                    'KKA2_KLEPN_Mikkelsen2014-Kan18_avg',
                    'BLAT_ECOLX_Ostermeier2014-linear',
                    'BLAT_ECOLX_Palzkill2012-ddG_stat',
                    'BLAT_ECOLX_Ranganathan2015-2500',
                    'B3VI55_LIPST_Klesmith2015_SelectionOne',
                    'B3VI55_LIPSTSTABLE_Klesmith2015_SelectionTwo',
                    'HSP82_YEAST_Bolon2016-selection_coefficient',
                    'AMIE_PSEAE_Wrenbeck2017_isobutyramide_normalized_fitness',
                    'MK01_HUMAN_Brenan2016_DOX_Average',
                    ]:
        print(dataset)
        for seed in range(10):
            print(seed)
            run_dataset(dataset, regressor,'vae+PST+esm1b',seed,'_emb')

    for dataset in [
                    # 'GB1_Olson2014_ddg',
                    # 'PABP_YEAST_Fields2013-linear',
                    # 'GFP_AEQVI_Sarkisyan2016'
                    ]:
        for seed in range(10):
            run_dataset(dataset,regressor, 'vae+PST+esm1b', seed, '_all_emb')
        # run_dataset(dataset, '_single_emb')
        # run_dataset(dataset, '_n_mut_2_emb')
        # run_dataset(dataset, '_n_mut_3_emb')
        # run_dataset(dataset, '_n_mut_4_emb')


    for dataset in [
                # 'IF1_ECOLI_Kelsic2016_fitness_rich_NMR',
                # 'UBE4B_MOUSE_Klevit2013-nscor_log2_ratio_single_NMR',
                # 'BRCA1_HUMAN_Fields2015-e3_NMR',
                # 'BRCA1_HUMAN_Fields2015-y2h_NMR',
                    ]:
        for seed in range(10):
            run_dataset_NMR(dataset,regressor, 'vae+PST+esm1b',seed,'_emb')


