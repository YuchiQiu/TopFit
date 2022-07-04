import pathlib
import torch
from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained
import numpy as np
import pandas as pd
import os
import pickle
from sklearn.preprocessing import StandardScaler
from filepath_dir import TAPE_PRETRAIN_PATH,TAPE_MODEL_LOCATIONS
import argparse


def compare_seq_files(data_path):
    data=pd.read_csv(os.path.join(data_path,'data.csv'))
    seq1=data['seq'].values
    seq_lines=open(os.path.join(data_path,'seqs.fasta')).readlines()
    seq2=[]
    for line in seq_lines:
        if not '>' in line:
            line = line.replace('\n', '')
            seq2.append(line)
    assert len(seq1)==len(seq2), 'number of sequences are different in .fasta and .csv files'
    for i in range(len(seq1)):
        assert seq1[i]==seq2[i],'different sequence got in .fasta and .csv'
    max_length=0
    for s in seq1:
        if len(s)>max_length:
            max_length=len(s)
    return data,max_length
def get_batch_fasta(seqs,dataset_name,model_name,batch_size):
    fasta_filenames=[]
    prefix=dataset_name+'_'+model_name+'_'
    index=np.arange(0,len(seqs)+1,batch_size)
    if index[-1]!=len(seqs):
        index=np.append(index,len(seqs))
    for id in range(1,len(index)):
        a=index[id-1]
        b=index[id]
        filename=prefix+str(id-1)+'.fasta'
        fasta_filenames.append(filename)
        file=open(filename,'w')
        for i in range(a,b):
            seq=seqs[i]
            file.write('>id_'+str(i)+'\n')
            file.write(seq+'\n')
        file.close()
    return fasta_filenames

def run_embedding(seqs, dataset_name,model_name,output_dir,max_length,batch_size=64):
    fasta_filenames = get_batch_fasta(seqs, dataset_name, model_name,batch_size)
    embedding_avg=[]
    weights_loc=TAPE_MODEL_LOCATIONS.get(model_name)

    for fasta_filename in fasta_filenames:
        output_file=dataset_name+'_'+model_name+'.pkl'
        os.system('tape-embed '+fasta_filename+' '+model_name+' '+' --load-from '+weights_loc+' --output '+output_file)

        with open(output_file, "rb") as f:
            raw_embeddings = pickle.load(f)
        for itm in raw_embeddings:
            embedding_avg.append(np.mean(itm,axis=1).reshape(-1))
        os.system('rm '+output_file)
        os.system('rm '+fasta_filename)
    embedding_avg=np.asarray(embedding_avg)
    file='tape_'+model_name+'.npy'
    np.save(os.path.join(output_dir, 'unnorm', file),embedding_avg)
    scaler = StandardScaler()
    embedding_avg = scaler.fit_transform(embedding_avg)
    np.save(os.path.join(output_dir, 'norm', file),embedding_avg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_name', type=str)
    parser.add_argument('--model', type=str,help='1. resnet; 2. bepler; 3. unirep; 4. transformer; 5. lstm',default='lstm')
    parser.add_argument('--output_dir', type=str,default='')
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()


    model_name=args.model
    dataset_name = args.dataset_name

    data_path=os.path.join('data',dataset_name)
    if args.output_dir=='':
        output_dir=pathlib.Path(os.path.join('Features',dataset_name))
    else:
        output_dir=pathlib.Path(args.output_dir)
    data,max_length=compare_seq_files(data_path)
    seqs = data['seq'].values
    seqs = [seq.replace('-', 'X') for seq in seqs]

    run_embedding(seqs, dataset_name,model_name,output_dir,max_length,batch_size=args.batch_size)


