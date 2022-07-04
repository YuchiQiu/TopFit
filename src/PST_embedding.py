import numpy as np
import os
import sys
import pandas as pd
import pickle
from pathlib import Path
from PST import runPST
import argparse
sys.path.insert(0, 'src/')
from filepath_dir import Chain_dir, Uniprot_to_PDB, dataset_to_uniprot
def get_upper(characters):
    characters_new=''
    for a in characters:
        if a.isalpha():
            a=a.upper()
        characters_new+=a
    return characters_new

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset',type=str)
    parser.add_argument('a',type=int,help="index for the first mutation to be calculated")
    parser.add_argument('b',type=int,help="index for the last mutation to be calculated",)
    parser.add_argument("--max_mut",type=int,help="max number of mutations",default=1)
    parser.add_argument("--site_method",type=str,help="Method for dealing with features involving multiple mutations. "
                                                      "`align` concatenate features to a vector, and use zero padding to fill position if number of mutations is fewer than --max_mut."
                                                      "`sum` sum up features over all mutational sites"
                                                      "Options: 1. align 2. sum 3. avg ",default='sum')
    parser.add_argument('--structure_method', type=str,default='')
    args = parser.parse_args()
    dataset=args.dataset
    a=args.a
    b=args.b
    structure_method=args.structure_method

    uniprot=dataset_to_uniprot.get(dataset)
    PDBid_list = Uniprot_to_PDB.get(uniprot)
    if structure_method in PDBid_list:
        PDBid=PDBid_list.get(structure_method)
    else:
        PDBid=PDBid_list.get('default')
    Chain = Chain_dir.get(uniprot)
    data = pd.read_csv('data/' + dataset + '/data.csv')
    pH = '7.0'
    for data_id in range(a,b):
        print(data_id)
        MUT=data['mutant'].values[data_id].split(',')
        mutations=[mut[0]+Chain+mut[1:] for mut in MUT]
        y = float(data['log_fitness'].values[data_id])
        num_sites = len(mutations)
        if not structure_method=='':
            data_path=uniprot+'_'+structure_method+'/'
        else:
            data_path=uniprot+'/'
        os.system('mkdir -p '+data_path)
        working_dir = data_path + '_'.join(mutations)+'/'
        if not os.path.exists(working_dir+'X_PH0dthmute.npy'):
            os.system('rm -r '+working_dir)
            os.system('mkdir -p ' + working_dir)
            print('running on >>>>>>>>>'+ working_dir)
            print(PDBid)
            if structure_method=='NMR':
                MODEL_ID=[]
                for i in range(100):
                    pdbfile='structure_data/'+uniprot+'/processed_PDB/'+PDBid+'_m'+str(i)+'.pdb'
                    if os.path.exists(pdbfile):
                        os.system('cp '+pdbfile+' '+working_dir+PDBid+'_m'+str(i)+'_WT.pdb')
                        MODEL_ID.append(i)
                home_dir=os.getcwd()
                os.chdir(working_dir)
                runPST.main_NMR(PDBid, Chain, mutations, num_sites, pH, ['WT','MT'],
                                     max_mut=args.max_mut,site_method=args.site_method,MODEL_ID=MODEL_ID)
            else:
                os.system('cp structure_data/'+uniprot+'/processed_PDB/'+PDBid+'.pdb '+working_dir+PDBid+'_WT.pdb')
                home_dir=os.getcwd()
                os.chdir(working_dir)
                runPST.main(PDBid, Chain, mutations, num_sites, pH, ['WT','MT'],
                                     max_mut=args.max_mut,site_method=args.site_method)
            os.system('rm *.pdb')

            os.chdir(home_dir)



