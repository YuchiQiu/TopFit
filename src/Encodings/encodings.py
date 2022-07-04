import os
import numpy as np
import pandas as pd
import random
from Bio.Align import substitution_matrices
from utils import seqs_to_onehot, seqs_to_georgiev,get_wt_seq, read_fasta, seq2effect
from utils import mutant2seq, load, load_rows_by_numbers
import utils

UNSUPERVISED_LIST=['vae','esm1b_pll','esm1v_pll','eunirep_pll']

def select_training_data(data, n_train, scores):
    sorted_idx = np.argsort(scores)
    idx = sorted_idx[-n_train:]
    return data.iloc[idx, :].sample(n=n_train)
# class BaseEncoding():
#     """Abstract class for predictors."""
#
#     def __init__(self, dataset_name, **kwargs):
#         self.dataset_name = dataset_name
#     def select_training_data(self, data, n_train,seed=0):
#         return data.sample(n=n_train,random_state=seed)


class BaseEncoding():

    def __init__(self, dataset_name,**kwargs):
        self.UNSUPERVISED_LIST = UNSUPERVISED_LIST
        self.dataset_name = dataset_name
    def select_training_data(self, data, n_train,seed=0):
        return data.sample(n=n_train,random_state=seed)
    def seq2feat(self, seqs):
        raise NotImplementedError


class JointEncoder(BaseEncoding):
    """Combining regression predictors by training jointly."""

    def  __init__(self, dataset_name, predictor_classes, predictor_name,**kwargs):
        super(JointEncoder, self).__init__(dataset_name, **kwargs)
        self.predictors = []
        for c, name in zip(predictor_classes, predictor_name):
            self.predictors.append(c(dataset_name,predictor_name=name, **kwargs))

    def seq2feat(self, seqs):
        # To apply different regularziation coefficients we scale the features
        # by a multiplier in Ridge regression
        """
        Obtain encodings for given sequences by loading existing files.

        Parameters
        ----------
        seqs: list of sequences (N sequences)

        Returns
        ----------
        embedding: NxL or NxCxL numpy array
            Encoding from sequence embedding.
            Including constant encoding (e.g. onehot), deep sequence model encoding (e.g. transformer), structure-based encoding (e.g. PST)
        unsupervised: NxL or NxCxL numpy array
            Encoding from probability models.
            Including deep sequence elbo (vae), etc.
        ensemble_structure: bool
            It can be True only when structure-based encoding is included
                and multiple structures are available.
                We concatenate features from multiple structures.

            If True: both `embedding` and `unsupervised` are 3D numpy array (NxCxL).
                N is number of sequences.
                C is number of structures considered for ensemble.
                    Usually used for NMR structure where multiple models are available
                L is the dimension of the encoding for each sequence
            If False: both `embedding` and `unsupervised` are 2D numpy array (NxL).
        """
        embedding=[]
        unsupervised=[]
        ensemble_structure=False
        nmr_id=[]
        for i,p in enumerate(self.predictors):
            f1,f2=p.seq2feat(seqs)
            embedding.append(f1)
            unsupervised.append(f2)
            if len(f1.shape)==3:
                ensemble_structure=True
                MODEL_NUM=f1.shape[1]
                nmr_id.append(i)
        if ensemble_structure:
            for i,f in enumerate(embedding):
                if len(f.shape)==2:
                    embedding[i]=np.repeat(f[:,np.newaxis,:],MODEL_NUM,axis=1)
            for i,f in enumerate(unsupervised):
                if len(f.shape)==2:
                    unsupervised[i]=np.repeat(f[:,np.newaxis,:],MODEL_NUM,axis=1)
            embedding=np.concatenate(embedding,axis=2)
            unsupervised=np.concatenate(unsupervised,axis=2)
        else:
            embedding=np.concatenate(embedding, axis=1)
            unsupervised=np.concatenate(unsupervised, axis=1)
        unsupervised=unsupervised
        return embedding,unsupervised,ensemble_structure




'''
Modules for embedding generations
'''
class Onehot(BaseEncoding):
    """Simple one hot encoding + ridge regression."""

    def __init__(self, dataset_name, predictor_name, **kwargs):
        super(Onehot, self).__init__(
                dataset_name, **kwargs)
        self.predictor_name=predictor_name

    def seq2feat(self, seqs):
        X=seqs_to_onehot(seqs)
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        if self.predictor_name in self.UNSUPERVISED_LIST:
            unsupervised = X
            embedding = np.zeros([unsupervised.shape[0], 0])
        else:
            embedding = X
            unsupervised = np.zeros([embedding.shape[0], 0])
        return embedding,unsupervised

class Georgiev(BaseEncoding):
    """Georgiev encoding + ridge regression."""

    def __init__(self, dataset_name,predictor_name, **kwargs):
        super(Georgiev, self).__init__(
                dataset_name, **kwargs)
        self.predictor_name=predictor_name

    def seq2feat(self, seqs):
        X=seqs_to_georgiev(seqs)
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        if self.predictor_name in self.UNSUPERVISED_LIST:
            unsupervised = X
            embedding = np.zeros([unsupervised.shape[0], 0])
        else:
            embedding = X
            unsupervised = np.zeros([embedding.shape[0], 0])
        return unsupervised,embedding

class LoadNumpy(BaseEncoding):
    """Regression on load precalculated numpy embedding."""

    def __init__(self, dataset_name, predictor_name, **kwargs):
        super(LoadNumpy, self).__init__(dataset_name,
                                                       **kwargs)
        self.load_rep(dataset_name, predictor_name)
        self.predictor_name=predictor_name
    def load_rep(self, dataset_name, predictor_name):
        self.feature_path = os.path.join('Features/', dataset_name,'norm/',
                predictor_name+'.npy')
        self.seq_path = os.path.join('data', dataset_name, 'seqs.fasta')
        self.seqs = read_fasta(self.seq_path)
        self.seq2id = dict(zip(self.seqs, range(len(self.seqs))))

    def seq2feat(self, seqs):
        """Look up representation by sequences."""
        ids = np.asarray([self.seq2id[s] for s in seqs])
        data=np.load(self.feature_path)
        X=data[ids,:]
        if len(X.shape)==1:
            X=X.reshape(-1,1)
        if self.predictor_name in self.UNSUPERVISED_LIST:
            unsupervised=X
            embedding=np.zeros([unsupervised.shape[0],0])
        else:
            embedding=X
            unsupervised=np.zeros([embedding.shape[0],0])

        return embedding,unsupervised

'''
Modules for zero-shot inference 
'''
class VaeElbo(BaseEncoding):
    "deepseq vae prediction."""

    def __init__(self, dataset_name,predictor_name, **kwargs):
        super(VaeElbo, self).__init__(dataset_name, **kwargs)
        self.predictor_name=predictor_name
        path = os.path.join('inference', dataset_name, 'elbo.npy')
        if os.path.exists(path):
            delta_elbo = np.loadtxt(path)
            seqs_path = os.path.join('data', dataset_name, 'seqs.fasta')
            # if not os.path.exists(seqs_path):
            #     seqs_path = os.path.join('data', dataset_name, 'seqs.fasta')
            seqs = read_fasta(seqs_path)
            assert len(delta_elbo) == len(seqs), 'file length mismatch'
            self.seq2score_dict = dict(zip(seqs, delta_elbo))
        else:
            df = pd.read_csv(os.path.join('inference', dataset_name,
                'vae_elbo.csv'))
            df = df[np.isfinite(df.mutation_effect_prediction_vae_ensemble)]
            wtseqs, wtids = read_fasta(os.path.join('data', dataset_name,
                'wt.fasta'), return_ids=True)
            offset = int(wtids[0].split('/')[-1].split('-')[0])
            wt = wtseqs[0]
            seqs = [mutant2seq(m, wt, offset) for m in df.mutant.values]
            self.seq2score_dict = dict(zip(seqs,
                df.mutation_effect_prediction_vae_ensemble))

    def seq2score(self, seqs):
        scores = np.array([self.seq2score_dict.get(s, 0.0) for s in seqs])
        #return np.nan_to_num(scores)
        return scores

    def seq2feat(self, seqs):
        X=self.seq2score(seqs)[:, None]
        if len(X.shape)==1:
            X=X.reshape(-1,1)
        if self.predictor_name in self.UNSUPERVISED_LIST:
            unsupervised=X
            embedding=np.zeros([unsupervised.shape[0],0])
        else:
            embedding=X
            unsupervised=np.zeros([embedding.shape[0],0])

        return embedding,unsupervised

    def predict_unsupervised(self, seqs):
        return self.seq2score(seqs)

class ESMPLL(BaseEncoding):
    """ESM likelihood prediction."""

    def __init__(self, dataset_name, predictor_name, path_prefix='',
            **kwargs):
        super(ESMPLL, self).__init__(dataset_name)
        self.predictor_name=predictor_name
        seqs_path = path_prefix + os.path.join('data', dataset_name, 'seqs.fasta')
        seqs = read_fasta(seqs_path)
        id2seq = pd.Series(index=np.arange(len(seqs)), data=seqs, name='seq')

        esm_data_path = path_prefix + os.path.join('inference', dataset_name,
                 predictor_name+ '.csv')
        ll = pd.read_csv(esm_data_path, index_col=0)
        ll['id'] = ll.index.to_series().apply(
                lambda x: int(x.replace('id_', '')))
        ll = ll.join(id2seq, on='id', how='left')
        self.seq2score_dict = dict(zip(ll.seq, ll.pll))

    def seq2score(self, seqs):
        scores = np.array([self.seq2score_dict.get(s, 0.0) for s in seqs])
        return scores

    def seq2feat(self, seqs):
        X=self.seq2score(seqs)[:, None]
        if len(X.shape)==1:
            X=X.reshape(-1,1)
        if self.predictor_name in self.UNSUPERVISED_LIST:
            unsupervised = X
            embedding = np.zeros([unsupervised.shape[0], 0])
        else:
            embedding = X
            unsupervised = np.zeros([embedding.shape[0], 0])

        return embedding,unsupervised
    def predict_unsupervised(self, seqs):
        return self.seq2score(seqs)

class EUniRepPLL(BaseEncoding):
    """UniRep likelihood prediction."""

    def __init__(self, dataset_name,predictor_name, **kwargs):
        super(EUniRepPLL, self).__init__(
            dataset_name)
        self.predictor_name=predictor_name
        self.seq_path = os.path.join('inference', dataset_name,
                predictor_name, f'seqs.npy')
        self.seqs = np.loadtxt(self.seq_path, dtype=str, delimiter=' ')
        self.seq2id = dict(zip(self.seqs, range(len(self.seqs))))
        self.loss_path = os.path.join('inference', dataset_name,
                predictor_name, f'loss.npy*')
    def seq2feat(self, seqs):
        """Look up log likelihood by sequence."""
        ids = [self.seq2id[s] for s in seqs]
        X=-load_rows_by_numbers(self.loss_path, ids)
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        if self.predictor_name in self.UNSUPERVISED_LIST:
            unsupervised = X
            embedding = np.zeros([unsupervised.shape[0], 0])
        else:
            embedding = X
            unsupervised = np.zeros([embedding.shape[0], 0])
        return embedding,unsupervised

    def predict_unsupervised(self, seqs):
        ids = [self.seq2id[s] for s in seqs]
        y=-load_rows_by_numbers(self.loss_path, ids)
        return y.ravel()


