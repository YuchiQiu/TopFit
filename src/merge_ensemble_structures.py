import numpy as np
import pandas as pd
from evaluate_singleprocess import summary_csv
import argparse
import os
from Encodings import get_encoder_cls, JointEncoder,get_encoder_names

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
    parser.add_argument('--n_train', type=int, default=96)
    parser.add_argument('--results_suffix', type=str, default='')
    parser.add_argument('--seed', type=int, default=0,
            help='random seed for training and testing data split')
    args = parser.parse_args()



    outdir = os.path.join('results', args.dataset_name)
    outpath = os.path.join(outdir, f'results{args.results_suffix}.csv')
    outpath = f'{outpath}-{os.getpid()}'  # each process writes to its own file

    encoders = get_encoder_names(args.encoder_name)
    structure_id=0
    result_file='results' + args.results_suffix + '_'+\
                args.encoder_name + '_n' + str(args.n_train)+\
                '_seed' + str(args.seed) + '_strc'+\
                str(structure_id) + '.npz'
    outpath_npz = os.path.join(outdir, result_file)

    DATA=np.load(outpath_npz,allow_pickle=True)
    preds=[]
    # np.savez(outpath_npz, dataset_name=dataset_name, encoder_name=encoder_name, n_train=n_train,
    #          seed=seed, data=data, test=test, max_n_mut=max_n_mut, regressor_dic=regressor_dic, outpath=outpath,
    #          n_structure=n_structure, structure_id=structure_id)
    n_structure=DATA['n_structure'].item()
    dataset_name=DATA['dataset_name'].item()
    encoder_name=DATA['encoder_name'].item()
    n_train=DATA['n_train'].item()
    seed=DATA['seed'].item()
    data_key=DATA['data_key']
    data=pd.DataFrame(data=DATA['data'],columns=data_key)

    test_key=DATA['test_key']
    test0=pd.DataFrame(data=DATA['test'],columns=test_key)
    preds.append(test0['pred'].values)
    max_n_mut=DATA['max_n_mut'].item()
    regressor_dic=DATA['regressor_dic'].item()
    if '_NMR' in dataset_name:
        regressor_dic['n_structures'] = n_structure
    results_suffix=DATA['results_suffix'].item()
    if n_structure>1:
        for structure_id in range(1,n_structure):
            print(structure_id)
            results_file='results' + results_suffix + '_'+\
                         encoder_name + '_n' + str(n_train)+\
                         '_seed' + str(seed) + '_strc'+\
                         str(structure_id) + '.npz'
            outpath_npz = os.path.join(outdir, results_file)
            DATA = np.load(outpath_npz, allow_pickle=True)
            tmp=pd.DataFrame(data=DATA['test'],columns=test_key)
            for i in range(len(test0)):
                assert tmp['seq'][i]==test0['seq'][i]
                assert tmp['log_fitness'][i]==test0['log_fitness'][i]
                assert tmp['n_mut'][i]==test0['n_mut'][i]
                assert tmp['mutant'][i]==test0['mutant'][i]
            preds.append(tmp['pred'].values)

    preds=np.asarray(preds)
    preds=np.mean(preds,axis=0)
    print(preds.shape)
    test0['pred']=preds
    summary_csv(dataset_name, encoder_name, n_train, seed, data, test0, max_n_mut, regressor_dic, outpath)
    for structure_id in range(n_structure):
        results_file='results' + results_suffix + '_'+\
                     encoder_name + '_n' + str(n_train)+\
                     '_seed' + str(seed) + '_strc'+\
                     str(structure_id) + '.npz'
        outpath_npz = os.path.join(outdir, results_file)
        os.system('rm '+outpath_npz)

if __name__ == '__main__':
    main()
