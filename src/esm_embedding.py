import pathlib
import torch
from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained
import numpy as np
import pandas as pd
import os
import argparse
from sklearn.preprocessing import StandardScaler

esm_dict = {
    'esm1v': 'esm1v_t33_650M_UR90S_1', # use first of 5 models
    'esm1b': 'esm1b_t33_650M_UR50S'}

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
def run_embedding(seqs, dataset_name,model_name,output_dir,max_length,batch_size=64):
    model, alphabet = pretrained.load_model_and_alphabet(esm_dict.get(model_name))
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
        print("Transferred model to GPU")
    batch_converter = alphabet.get_batch_converter()
    embedding=[]
    n_batches = int(len(seqs) / batch_size)
    leftover = len(seqs) % batch_size
    n_batches += int(bool(leftover))
    for i in range(n_batches):
        if i == n_batches - 1:
            batch_seqs = seqs[-leftover:]
        else:
            batch_seqs = seqs[i * batch_size:(i + 1) * batch_size]
        batch=[(str(i*batch_size+k),s) for k,s in enumerate(batch_seqs)]
        _, _, batch_tokens = batch_converter(batch)
        if torch.cuda.is_available():
            batch_tokens=batch_tokens.to(device='cuda')
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[len(model.layers)],return_contacts=False)
        token_representations = results["representations"][33].cpu().numpy()
        for k,s in enumerate(batch_seqs):
            tokens=token_representations[k, 1 : len(s) + 1,:].reshape([len(s),token_representations.shape[2]])
            tokens_mean=np.mean(tokens,axis=0)
            embedding.append(tokens_mean)

    embedding=np.asarray(embedding)
    file=model_name+'.npy'
    np.save(output_dir/ 'unnorm'/ file,embedding)
    scaler = StandardScaler()
    embedding = scaler.fit_transform(embedding)
    np.save(output_dir/ 'norm'/ file,embedding)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_name', type=str)
    parser.add_argument("--model",type=str,help="model, options: 1. esm1b; 2. esm1v",default="esm1b")
    parser.add_argument('--output_dir', type=str,default='')
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    model=args.model
    dataset_name = args.dataset_name

    data_path=os.path.join('data',dataset_name)
    if args.output_dir=='':
        output_dir=pathlib.Path(os.path.join('Features',dataset_name))
    else:
        output_dir=pathlib.Path(args.output_dir)
    data,max_length=compare_seq_files(data_path)
    pathlib.Path(output_dir / 'norm').mkdir(parents=True, exist_ok=True)
    pathlib.Path(output_dir / 'unnorm').mkdir(parents=True, exist_ok=True)


    seqs = data['seq'].values
    seqs = [seq.replace('-', 'X') for seq in seqs]

    run_embedding(seqs,dataset_name, model,output_dir,max_length,batch_size=args.batch_size)


