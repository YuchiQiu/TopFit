import numpy as np
import os
import sys
import pandas as pd
from filepath_dir import Uniprot_to_PDB,Chain_dir,dataset_to_uniprot
from sklearn.preprocessing import StandardScaler


def collect_feature(data_path,dataset,FeatureName,data,Chain,structure_method):
    Feature=[]
    mutations=data['mutant'].values
    for MUT in mutations[0:4]:
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



if __name__ == "__main__":
    #dataset = 'BLAT_ECOLX_Ostermeier2014-linear'
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset',type=str)
    parser.add_argument('--structure_method', type=str,default='')
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
        os.system('mkdir Features/'+dataset+'_'+structure_method)
    else:
        data_path = uniprot + '/'
        os.system('mkdir Features/'+dataset)

    FeatureList = ['PH','PST']
    for FeatureName in FeatureList:
        collect_feature(data_path,dataset,FeatureName,data,Chain,structure_method)

