import argparse
import os
import pathlib

import numpy as np
import pandas as pd
import tensorflow as tf

from unirep import babbler1900
from utils import load_and_filter_seqs, format_batch_seqs, nonpad_len
from esm_embedding import compare_seq_files
from sklearn.preprocessing import StandardScaler
from filepath_dir import dataset_to_uniprot,UNIREP_PATH
def run_embedding(seqs, dataset,model, output_dir,
        batch_size=64):
    uniprot_id=dataset_to_uniprot.get(dataset)
    if len(seqs) < batch_size:
        batch_size = len(seqs)
    if model=='eunirep':
        model_weight_path=os.path.join(UNIREP_PATH,uniprot_id)
    elif model=='gunirep':
        model_weight_path=os.path.join(UNIREP_PATH,'global')
    babbler_class = babbler1900
    # Load model weights
    b = babbler_class(batch_size=batch_size, model_path=model_weight_path)
    # Load ops
    final_hidden_op, avg_hidden_op, x_ph, batch_size_ph, seq_len_ph, init_state_ph = b.get_rep_ops()
    logits_op, loss_op, x_ph, y_ph, batch_size_ph, init_state_ph = b.get_babbler_ops()
    batch_loss_op = b.batch_losses

    final_hidden_vals = []
    avg_hidden_vals = []
    loss_vals = []
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        n_batches = int(len(seqs) / batch_size)
        leftover = len(seqs) % batch_size
        n_batches += int(bool(leftover))
        for i in range(n_batches):
            print('----Running inference for batch # %d------' % i)
            if i == n_batches - 1:
                batch_seqs = seqs[-batch_size:]
            else:
                batch_seqs = seqs[i*batch_size:(i+1)*batch_size]
            batch = format_batch_seqs(batch_seqs)
            length = nonpad_len(batch)
            # Run final hidden op
            avg_hidden_, loss_ = sess.run(
                [avg_hidden_op, batch_loss_op],
                feed_dict={
                    # Important! Shift input and expected target by 1.
                    x_ph: batch[:, :-1],
                    y_ph: batch[:, 1:],
                    batch_size_ph: batch.shape[0],
                    seq_len_ph: length,
                    init_state_ph:b._zero_state
                })
            if i == n_batches - 1:
                # loss_vals.append(loss_[-leftover:])
                avg_hidden_vals.append(avg_hidden_[-leftover:])
            else:
                # loss_vals.append(loss_)
                avg_hidden_vals.append(avg_hidden_)

    avg_hidden_vals = np.concatenate(avg_hidden_vals, axis=0)
    avg_hidden_filename = os.path.join(
            output_dir, 'unnorm',model+'.npy')
    np.save(avg_hidden_filename, avg_hidden_vals)
    scaler = StandardScaler()
    avg_hidden_vals = scaler.fit_transform(avg_hidden_vals)
    avg_hidden_filename = os.path.join(
            output_dir, 'norm',model+'.npy')
    np.save(avg_hidden_filename, avg_hidden_vals)
    print('Ran inference on %d sequences. Saved results to %s.' %
            (len(seqs), args.output_dir))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str)
    parser.add_argument('--model', type=str,help='1. eunirep: fine-tune unirep; 2. gunirep: global unirep',default='gunirep')
    parser.add_argument('--output_dir', type=str,default='')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--save_hidden', dest='save_hidden', action='store_true')
    args = parser.parse_args()

    dataset=args.dataset
    data_path=os.path.join('data',dataset)
    if args.output_dir=='':
        output_dir=os.path.join('Features',dataset)
    else:
        output_dir=args.output_dir
    data,max_length=compare_seq_files(data_path)
    pathlib.Path(output_dir + '/norm').mkdir(parents=True, exist_ok=True)
    pathlib.Path(output_dir + '/unnorm').mkdir(parents=True, exist_ok=True)
    seqs = data['seq'].values
    seqs = [seq.replace('-', 'X') for seq in seqs]
    model=args.model
    run_embedding(seqs, dataset,model,
            output_dir, batch_size=args.batch_size)
