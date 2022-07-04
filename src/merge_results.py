import pandas as pd
from utils import merge_dfs
import os,sys

# dataset_name=sys.argv[1]
# results_suffix=sys.argv[2]
def run_dataset(dataset_name,results_suffix):
    outdir = os.path.join('results', dataset_name)
    print(dataset_name+results_suffix)
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    outpath = os.path.join(outdir, f'results{results_suffix}.csv')
    merge_dfs(f'{outpath}*', outpath,
              index_cols=['dataset', 'encoder', 'seed', 'n_train'],
              groupby_cols=['encoder', 'encoder_params', 'n_train', 'topk'],
              ignore_cols=['seed'])
    if os.path.exists(outpath):
        data=pd.read_csv(outpath)
        print(len(data))



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset',type=str)
    parser.add_argument('--results_suffix', type=str,default='')
    args = parser.parse_args()
    run_dataset(args.dataset,args.results_suffix)


