import numpy as np
import os
import sys
import pandas as pd
from filepath_dir import Uniprot_to_PDB,Chain_dir,dataset_to_uniprot
from sklearn.preprocessing import StandardScaler


def collect_feature(data_path,dataset,FeatureName,data,Chain,structure_method):
    Feature=[]
    mutations=data['mutant'].values
    for MUT in mutations:
        MUT=MUT.split(',')
        mut_path=[mut[0] + Chain + mut[1:] for mut in MUT]
        file_dir = data_path + '_'.join(mut_path)+'/'
        XX=np.load(file_dir+'X_'+FeatureName+'.npy')
        Feature.append(XX)

    Feature=np.asarray(Feature)
    if len(Feature.shape)==2:
        scaler = StandardScaler()
        Feature_normalized=scaler.fit_transform(Feature)
    else:
        scalers = {}
        Feature_normalized=np.zeros(Feature.shape)
        for i in range(Feature.shape[1]):
            scalers[i] = StandardScaler()
            Feature_normalized[:, i, :] = scalers[i].fit_transform(Feature[:, i, :])
        # Feature=np.swapaxes(Feature, 0, 1)
        # Feature_normalized=np.swapaxes(Feature_normalized, 0, 1)
    print(Feature.shape)
    if not structure_method=='':
        dataset=dataset+'_'+structure_method
    os.system('mkdir -p Features/'+dataset+'/unnorm/')
    os.system('mkdir Features/'+dataset+'/norm/')
    np.save('Features/'+dataset+'/unnorm/'+FeatureName + '.npy',Feature)
    np.save('Features/'+dataset+'/norm/'+FeatureName + '.npy',Feature_normalized)

def combine_feature(FeatureList,dataset,OutputName,normalized,structure_method):
    if not structure_method=='':
        dataset=dataset+'_'+structure_method
    Feature=np.load('Features/'+dataset+'/'+normalized+'/'+FeatureList[0]+'.npy')

    for id in range(1,len(FeatureList)):
        FeatureName=FeatureList[id]
        Feature=np.append(Feature,np.load('Features/'+dataset+'/'+normalized+'/'+FeatureName+'.npy'),axis=len(Feature.shape)-1)
    print(Feature.shape)
    np.save('Features/'+dataset+'/'+normalized+'/'+OutputName+'.npy',Feature)


if __name__ == "__main__":
    #dataset = 'BLAT_ECOLX_Ostermeier2014-linear'
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset',type=str)
    parser.add_argument('--structure_method', type=str,default='')
    parser.add_argument('--feature_list', type=str,default='PH+PST',help='A list of features. '
                                                                     'Multiple features can be input '
                                                                     'Seperated by "+" sign '
                                                                     'Available features:'
                                                                                '1. PH'
                                                                                '2. PST (PH12 features, using statistics on persistent bar'
                                                                                '3. PH12_landscape (PH12 features vectorized by Persistent Landscape'
                                                                                '4. PST12 (PST12 features for betti numbers, and non-harmonic spectra)' )
    args = parser.parse_args()
    dataset=args.dataset
    structure_method=args.structure_method

    uniprot=dataset_to_uniprot.get(dataset)
    PDBid_list = Uniprot_to_PDB.get(uniprot)
    if structure_method in PDBid_list:
        PDBid=PDBid_list.get(structure_method)
    else:
        PDBid=PDBid_list.get('default')
    Chain = Chain_dir.get(uniprot)

    data = pd.read_csv('data/' + dataset + '/data.csv')
    if not structure_method == '':
        data_path = uniprot + '_' + structure_method + '/'
        os.system('mkdir -p Features/'+dataset+'_'+structure_method)
    else:
        data_path = uniprot + '/'
        os.system('mkdir -p Features/'+dataset)
    feature_list = args.feature_list.split('+')
    FeatureList = ['PH','PST']
    for FeatureName in feature_list:
        if FeatureName =='PH':
            combine_list=['PH0mute', 'PH12mute']
            for featurename in combine_list:
                collect_feature(data_path, dataset, featurename, data, Chain, structure_method)
            for normalized in ['norm', 'unnorm']:
                combine_feature(combine_list, dataset, FeatureName, normalized, structure_method)
        elif FeatureName=='PST':
            combine_list=['PSTmutePH0', 'PH12mute', 'PSTmute']
            for featurename in combine_list:
                collect_feature(data_path, dataset, featurename, data, Chain, structure_method)
            for normalized in ['norm', 'unnorm']:
                combine_feature(combine_list, dataset, FeatureName, normalized, structure_method)
        else:
            collect_feature(data_path,dataset,FeatureName,data,Chain,structure_method)

