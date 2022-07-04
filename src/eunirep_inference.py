'''
Infers log-likelihoods from UniRep or eUniRep models.
codes are rewritten from https://github.com/chloechsu/combining-evolutionary-and-assay-labelled-data
'''

import argparse
import os
import pathlib

import numpy as np
import pandas as pd
import tensorflow as tf

from unirep import babbler1900
from utils import load_and_filter_seqs, save, format_batch_seqs, nonpad_len
from filepath_dir import dataset_to_uniprot,UNIREP_PATH

def run_inference(seqs, dataset,model, output_dir,
        batch_size=64, save_hidden=False):
    if len(seqs) < batch_size:
        batch_size = len(seqs)
    babbler_class = babbler1900
    uniprot_id=dataset_to_uniprot.get(dataset)
    # Load model weights
    if model=='eunirep':
        model_weight_path=os.path.join(UNIREP_PATH,uniprot_id)
    elif model=='gunirep':
        model_weight_path=os.path.join(UNIREP_PATH,'global')
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
            batch_seqs = [seq.replace('-', 'X') for seq in batch_seqs]
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
                loss_vals.append(loss_[-leftover:])
                if save_hidden:
                    avg_hidden_vals.append(avg_hidden_[-leftover:])
            else:
                loss_vals.append(loss_)
                if save_hidden:
                    avg_hidden_vals.append(avg_hidden_)

    loss_vals = np.concatenate(loss_vals, axis=0)
    loss_filename = os.path.join(
            output_dir, f'loss.npy')
    save(loss_filename, loss_vals)

    if save_hidden:
        avg_hidden_vals = np.concatenate(avg_hidden_vals, axis=0)
        avg_hidden_filename = os.path.join(
                output_dir, f'avg_hidden.npy')
        save(avg_hidden_filename, avg_hidden_vals)

    print('Ran inference on %d sequences. Saved results to %s.' %
            (len(seqs), output_dir))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('model_path', type=str)
    # parser.add_argument('data_path', type=str)
    # parser.add_argument('output_dir', type=str)
    parser.add_argument('dataset', type=str)
    parser.add_argument('--model', type=str,help='1. eunirep: fine-tune unirep; 2. gunirep: global unirep',default='gunirep')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--save_hidden', dest='save_hidden', action='store_true')
    args = parser.parse_args()

    uniprot=dataset_to_uniprot.get(args.dataset)
    data_path = 'data/'+ args.dataset+'/data.csv'
    model_path=UNIREP_PATH+uniprot+'/'
    output_dir='inference/'+args.dataset+'/eunirep_pll/'

    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    # print(data_path)
    seqs = load_and_filter_seqs(data_path)
    np.savetxt(os.path.join(output_dir, 'seqs.npy'), seqs, '%s')

    run_inference(seqs, args.dataset,args.model,
            output_dir, batch_size=args.batch_size,
            save_hidden=args.save_hidden)
