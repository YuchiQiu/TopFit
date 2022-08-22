# TopFit: Topology-offered protein fitness
**This is the source code of paper: _"TopFit: Topology-offered protein fitness" by Yuchi Qiu, and Guo-Wei Wei._ This work is under peer review.**

TopFit is a persistent spectral theory (PST)-based machine learning model that navigates protein fitness landscape. It integrates with sequence-based features and it is equipped with an ensemble regression. The PST-based features capture the mutation-induced local 3D structure changes. Whereas sequence-based models can deal with more general cases without resorting to 3D structures. The ensemble regression integrates various regression models automatically optimized by Bayesian optimization to have strong generalization for various sizes of training data.

# Table of Contents  

- [Installment](#installment)
- [Usage](#usage)
  * [PST embedding](#PST-embedding)
  * [Sequence-based embedding and evolutionary score](#Sequence-based)
  * [Ensemble regression](#Ensemble-regression)
  * [Running customized data](#Running-customized-data)
- [Sources](#sources) 
- [Reference](#reference) 

# Installment

The major modules of TopFit are implemented in Python3.6.6 in virtualenv with pacakges:
1. pytorch=1.10.0
2. [scikit-learn](https://scikit-learn.org/stable/)=0.23.1
3. [xgboost](https://xgboost.readthedocs.io/en/stable/)=1.5.0
4. scipy=1.4.1
6. biopython=1.79
7. numpy=1.19.5
8. h5py=2.10.0
9. [hyperopt](http://hyperopt.github.io/hyperopt/)=0.2.2
10. pandas=1.1.2
11. pickle=4.0
12. tqdm=4.32.1
13. [gudhi](https://gudhi.inria.fr)=3.3.0

The above pacakges are required by modules including PST, TAPE, ESM, and ensemble regression. The installation time is approaximately a few minutes.

## Additional pacakges

In additional to packages listed above, modules below need extra softwares or packages:
### PST module
PST module requires [VMD](https://www.ks.uiuc.edu/Research/vmd/) and [Jackal](http://honig.c2b2.columbia.edu/jackal) for structure data processing. Please install them and add the paths of their excutable files in `software_path()` function in `src/filepath_dir.py`.
### TAPE module
Follow the instruction to install [TAPE](https://github.com/songlab-cal/tape),and download pretrain weights. tensorflow=1.13.0 also needs to be installed. Please revise the path of its pretrain parameters in `TAPE_MODEL_LOCATIONS` in `src/filepath_dir.py`, its default value is `tape-neurips2019/pretrained_models/`.
### ESM module
Follow the instruction to install [ESM](https://github.com/facebookresearch/esm), and download pretrain weights for both esm1b and esm1v models. 

## Conda environments for two special sequence-based modules
The two modules below require different environment than above. Please create the conda environment below when using them. 

1. [DeepSequence VAE](https://github.com/debbiemarkslab/DeepSequence) is implemented by python2.7. Built the conda environment for it: `conda env create -f deep_sequence.yml`. Also follow the instruction to install this repo, and write the repo path in variable `WORKING_DIR` in both `src/vae_train.py` and `src/vae_inference.py`. This package is implemented by THEANO, please set up the CUDA environment for it to accelerate the VAE model training.
2. [eUniRep](https://github.com/churchlab/UniRep#obtaining-weight-files) and UniRep: we implemented it using codes from this [work](https://github.com/chloechsu/combining-evolutionary-and-assay-labelled-data) using tensorflow2.2.0. Build the conda environment: `conda env create -f protein_fitness_prediction.yml`. Please revise the path of its pretrain parameters in `UNIREP_PATH` in `src/filepath_dir.py`, its default value is `unirep_weights/`

# Usage
## PST embedding
First, run `src/PST_embedding.py $dataset $a $b` to generate embedding for dataset given in `data/$dataset/data.csv`. The first parameter `$dataset` is name of dataset. The second `$a` and third parameter `$b` give the range of mutational entries to run.

For example, run 
```
dataset=YAP1_HUMAN_Fields2012-singles-linear 
a=0 
b=319
python3 src/PST_embedding.py $dataset $a $b
```
The example run PST on entire dataset with index from 0 to 319. The range can be broken down to run one entry a time for parallelization by taking `a=n` and `b=n+1` for each job. Usually, one entry takes a few seconds to run. After `PST/generatePST.py` traverse all entries in the dataset, run 
```
dataset=YAP1_HUMAN_Fields2012-singles-linear
python src/merge_PST.py $dataset
``` 
to collect feature matrix for the dataset in dimension $N\times L$, where $N$ is the number of entries in the dataset and $L$ is the feature dimension. The feature matrix is stored in `Features/$dataset/unnorm/`, and the standardized features using `StandardScaler()` in [scikit-learn](https://scikit-learn.org/stable/) is stored in `Features/$dataset/norm/`. The regression model uses standardized features in default. `PST.npy` and `PH.npy` are for persistent spectral theory and persistent homology features, respectively. 

## Sequence-based embedding and evolutionary scores

The embedding feature matrix is stored in `Features/$dataset/unnorm/`, and its standarized matrix is stored in `Features/$dataset/norm/`. The embedding matrix has dimension $N\times L$, where $N$ is the number of entries in the dataset and $L$ is the feature dimension. The evolutionary scores are stored in `inference/$dataset/`. Each entry in the dataset has a score. Many functions for this module were rewritten from this [work](https://github.com/chloechsu/combining-evolutionary-and-assay-labelled-data)

### ESM model
Please install [ESM](https://github.com/facebookresearch/esm) and download the pretrained model weights. We implemented both esm1b and esm1v models. The esm1v provides five models from different random seeds, and we picked the first model. The embedding generation can be run, for example:
```
python src/esm_embedding.py $dataset $model
```
`$model` can be either `esm1b` or `esm1v`. The output file is stored in `Features/$dataset/norm/$model.npy`

The evoultionary score can be run:
```
python src/esm_inference.py $dataset --model $model
```
The output file is stored in `inference/$dataset/$model_pll.csv`

### TAPE models
tensorflow=1.13.0 needs to be installed. Please install [TAPE](https://github.com/songlab-cal/tape) and download the pretrained model weights.

The embedding can be run:
```
model=lstm
python src/tape_embedding.py $dataset --model $model
```
Five models are implemented in TAPE for `$model`: `resnet`, `bepler`, `unirep`, `transformer`, `lstm`. The file name is `tape_$model.npy`

The UniRep is implemented in TAPE also the UniRep package below. We generate both of them. But we only test the original UniRep embedding in this work.

### eUniRep and UniRep
The embedding can be run:
```
python src/eunirep_embedding.py $dataset --model $model
```
For UniRep, take `model=gunirep`. For eUniRep, take `model=eunirep`. 

The evolutionary score can be run, for example, 
```
python src/eunirep_inference.py $dataset --model $model
```
For UniRep, take `model=gunirep`. For eUniRep, take `model=eunirep`. 

The fine-tune eUniRep model is trained on MSA of the target protein. The pretrain weights are stored in `unirep_weights/$uniprot/` after training. To train it:run
```
uniprot=YAP1_HUMAN
seqs_fasta_path=alignments/$uniprot.a2m
save_weights_dir=unirep_weights/$uniprot
python src/unirep_evotune.py seqs_fasta_path save_weights_dir
```
`$seqs_fasta_path` is the MSA file in `.a2m` format. `$save_weights_dir` is the directory to save the pretrain weights. NOTICE: the fine-tune process is very slow and require GPU and huge memory. We run it on a NVDIA-v100 GPU node, allocate 100G memory, and a 2 days limits for the job. 

### DeepSequence VAE

The VAE is trained on MSA of the target protein by running:
```
uniprot=YAP1_HUMAN
THEANO_FLAGS='floatX=float32,device=cuda' python src/vae_train.py alignments/"$uniprot".a2m "$uniprot"_seed$seed $seed
```
The pretrained weights are stored in `params/` located in the directory of DeepSequence repo.

To obtained the evolutionary score, run:
```
dataset=YAP1_HUMAN_Fields2012-singles-linear
uniprot=YAP1_HUMAN
THEANO_FLAGS='floatX=float32,device=cuda' python src/vae_inference.py $uniprot data/"$dataset"/seqs.fasta data/"$dataset"/wt.fasta inference/"$dataset"/ --seed $seed
```
`$seed` is the random seed. We run 5 random seed with values `1,2,3,4,5`. The resulting evolutionary scores are stored in `inference/$dataset/` with name `elbo_seed$seed.npy`. After obtained scores from all random seeds, run `python src/merge_elbo.py` to average the predictions from all seeds and saved in `elbo.npy`. 

In addition, evolutionary scores for the most of datasets can be obtained from DeepSequence work directly. They are stored in `inference/$dataset/vae_elbo.csv`.

## Ensemble regression

The ensemble regression integrates multiple regressors with optimized hyperparameters. The top performing models are selected and their predictions are averaged to get the ensemble. To run the regression, first prepare the list of models. An example with full list of available models can be found in `Inputs/RegressorPara.csv`. Model type `ANN_deep` has more hidden units at each layer than model type `ANN`. Demo using ridge regression is `Inputs/RegressorPara_Ridge.csv`. The implementation of ensembel regression was rewritten from the codes in [MLDE](https://github.com/fhalab/MLDE) package. 

For example, run a Demo (running within 5 mintues) as the following: 
```
dataset=YAP1_HUMAN_Fields2012-singles-linear
encoder=vae+PST+esm1b
n_train=240
reg_para=Inputs/RegressorPara_Ridge.csv
python src/evaluate_singleprocess.py $dataset $encoder --seed $seed --n_train $n_train --reg_para $reg_para 
```
`$encoder` is the feature name. The different types of features are seperated by `+`. For example, `vae+PST+esm1b` is the feature combining `vae`, `PST`, and `esm1b`. `$seed` is the random seed for train/test set split. `$n_train` is the number of training data, `-1` indicates 80/20 split. `--reg_para` is the file for model list. 

To run 5-fold cross validation, use `src/evaluate_cv5.py` instead. 

### Output:
There are two types of output. 
1. The result is saved in directory `results/$dataset/` for various evaluating metrics including spearman and NDCG. Results from different seeds are saved in distinct `.csv` files with unique ID. It avoids conflicts when one tries to run multiple seeds in parallel. After all seeds are finished, run `src/merge_results.py $dataset` to merge all result files into a unique `.csv` file.
2. The other output can be a point-wise predictions on each single entry saved in `.npy` format. Need to use the `--save_pred` option to get the detailed predictions for each entry in the testing set. The ensemble regression on NMR can average the predictions from multiple NMR models. For detailed descriptions please run
```
python src/evaluate_singleprocess.py --help
```

## Runing customized data
Users can create their own data to run. Please follow the steps to create necessary inputs. 

1. Find the UniProtID (`$uniprot`), a high quality structure data (`$PDB`) of the target dataset (`$dataset`). Determine the chain name for the corresponding sequence (`$chain`). Find the resolution of the PDB data (`$resolution`) and the experimental method (`$method`). If it is a NMR structure, find the number of structure models (`$n_structure`) given in the NMR data. Then add a new entry in several dictionaries in `src/filepath_dir.py`:
 - `Chain_dir`: `key=$uniprot`, `value=$chain`.
 - `Uniprot_to_PDB`: `key=$uniprot`, `value=$PDB`.
 - `dataset_to_uniprot`: `key=$dataset`, `value=$uniprot`
 - `STUCTURE_TYPE`: `key=$dataset`, `value=$method`
 - `STUCTURE_RESOLUTION`: `key=$dataset`, `value=$resolution`
 - (if `$method` is NMR) `dataset_to_n_structure`: `key=$dataset`, `value=$n_structure`
2. Create sequence data in `data/$dataset/`. Refer to the format in the given sample
 - `data.csv`: DMS dataset (wildtype sequence is excluded) with columns: 
   - `seq` sequence for each mutation. 
   - `log_fitness`: fitness value. 
   - `n_mut`: number of mutational sites. 
   - `mutant`: mutation information is sperated by comma (`,`), for example, `T25R,E56L` means there are two mutational sites, the first mutation is on residue 25 with wildtype residue T and mutant residue R, the second mutation is on residue 56 with wildtype residue E and mutant residue L.
 -`seqs.fasta`: the list of sequences in the order of `seq` column in `data.csv` using fasta format. For each sequence, it takes two lines: first line uses format `>id_$id` (`$id` is the index for the sequence) and second line is sequence.
 - `wt.fasta`: wildtype sequence in fasta format.
3. Create optimized structure data in `structure_dataset/$dataset/processed_PDB/`.
 - First, download the PDB file (`$PDB.pdb` file) from PDB database or alphafold to `structure_dataset/$dataset/original_PDB/`. And add the `wt.fasta` for wildtype sequence and `$PDB.fasta` sequence file of the download PDB entry to `original_PDB/`. 
 - Use Jackal and VMD to optimize and revise the structure. Especially, when the `wt.fasta` and `$PDB.fasta` are not identical, the structure data needs to be mutated to match the wildtype sequence. Follow the manuscript description about the structure optimization. The precise processing functions are provided for each dataset: `structure_dataset/$dataset/get_PDB.py`. Please refer to our paper for the structure optimization details.
4. Create alignment file stored in `alignments/$dataset/`. We use [EVcoupling](https://v2.evcouplings.org) server to generate `.a2m` MSA file. We search over `UniRef100` database. The `Bitscore` is adjusted to ensure at least 70% residues are covered. See the manuscript or the [DeepSequence paper](https://doi.org/10.1038/s41592-018-0138-4) for details of MSA creation. 
5. The pretrain weights for fine-tune eUniRep models are stored in `unirep_weights/$uniprot/`. PST, PH and sequence-based embedding are all stored in `Features/$dataset/`. The evolutionary scores are 
## Data
### Demo dataset 
Run `./download_data_demo.sh` to download input and output data for `YAP1_HUMAN` dataset. 
### Source data
All source data are available in this [link](https://weilab.math.msu.edu/Downloads/TopFit/). `Data.tar.gz` contains sequence data, structure data, evolutionary scores, and alignment files for all datasets. `Features.tar.gz` contains normalized embedding features for all datasets. `unirep_weights.tar.gz` contains pretrain and fine-tune weights for eUniRep model.

# Reference
[1] This work "TopFit: Topology-offered protein fitness", by Yuchi Qiu and Guo-Wei Wei is under review.\
[2] [Learning protein fitness models from evolutionary and assay-labelled data, Nature Biotechnology 2022](https://doi.org/10.1038/s41587-021-01146-5) (Model integrates multiple sequence-based features)\
[3] [Informed training set design enables efficient machine learning-assisted directed protein evolution, Cell Systems 2021](https://doi.org/10.1016/j.cels.2021.07.008) (A work uses ensemble regression)\
[4] [Deep generative models of genetic variation capture the effects of mutations, Nature Methods, 2018](https://doi.org/10.1038/s41592-018-0138-4) (DeepSequence VAE)\
[5] [Low-N protein engineering with data-efficient deep learning, Nature Methods, 2021](https://doi.org/10.1038/s41592-021-01100-y) (eUniRep model)\
[6] [Evaluating Protein Transfer Learning with TAPE, Arxiv 2019](https://arxiv.org/abs/1906.08230) (TAPE model)\
[7] [Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences, PNAS 2021](https://doi.org/10.1073/pnas.2016239118) (ESM1b Transformer)\
[8] [Language models enable zero-shot prediction of the effects of mutations on protein function, BioRxiv, 2021](https://doi.org/10.1101/2021.07.09.450648) (ESM1v Transformer)

