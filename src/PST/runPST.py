import numpy as np
from .PST_features import *
from .PST_structure import *
def get_features(pro_mt,feature,site_method,max_mut,persistence=0):
    """""
        get features for one genotype 
    """""
    if feature=='PH0':
        mute0_dth, mute0_bar = PH_mute0(pro_mt)
        PH0_mutation = merge_features_site(mute0_bar,site_method=site_method,max_mut=max_mut)
        PH0dth_mutation = merge_features_site(mute0_dth,site_method=site_method,max_mut=max_mut)
        print('PH0 done')
        return PH0_mutation,PH0dth_mutation
    elif feature=='PH12':
        mute12_a, mute12_all = PH_mute12(pro_mt)
        PH12_mutation = merge_features_site(mute12_a, mute12_all,site_method=site_method,max_mut=max_mut)
        print('PH12 done')
        return PH12_mutation,None
    elif feature=='PST0':
        mute0_betti0, mute0_nonzero=PST_mute(pro_mt)
        PST_mutation_PH0=merge_features_site(mute0_betti0,site_method=site_method,max_mut=max_mut)
        PST_mutation0 = merge_features_site(mute0_nonzero,site_method=site_method,max_mut=max_mut)
        print('PST done')
        return PST_mutation_PH0,PST_mutation0
    elif feature=='PH12_landscape':
        mute12_a_landscape, mute12_all_landscape = PH_mute_landscape(pro_mt)
        PH12_mutation_landscape = merge_features_site(mute12_a_landscape, mute12_all_landscape
                                            ,site_method=site_method,max_mut=max_mut)
        print('PH12_landscape done')
        return PH12_mutation_landscape, None
    elif feature=='PST12':
        alpha_PST12_betti, alpha_PST12_nonzero=PST_L1L2_mute(pro_mt,persistence=persistence)
        PST12_betti_mutation = merge_features_site(alpha_PST12_betti,
                                           site_method=site_method,max_mut=max_mut)
        PST12_nonzero_mutation = merge_features_site(alpha_PST12_nonzero,
                                           site_method=site_method,max_mut=max_mut)
        print('PST12 done')
        return PST12_betti_mutation,PST12_nonzero_mutation
    # return PH0_mutation,PH0dth_mutation,PH12_mutation,PST_mutation_PH0,PST_mutation0

FEATURE_FILE_NAME={'PH0':['PH0mute','PH0dthmute'],
              'PH12':['PH12mute'],
              'PST0':['PSTmutePH0','PSTmute'],
              'PH12_landscape':['PH12mute_landscape'],
              'PST12':['PST12mutePH12','PST12mute'],#betti number; non-harmonic spectra
}
def save_feature(pro_wt,pro_mt,feature,site_method,max_mut=1,save_flg=True,persistence=0):
    # for feature in feature_list:
    a_mt, b_mt = get_features(pro_mt, feature, site_method=site_method,
                              max_mut=max_mut,persistence=persistence)
    a_wt, b_wt = get_features(pro_wt, feature, site_method=site_method,
                              max_mut=max_mut,persistence=persistence)
    Feature1=construct_features_MT_WT(a_mt, a_wt)
    Feature2=construct_features_MT_WT(b_mt, b_wt)
    files_name=FEATURE_FILE_NAME.get(feature)
    if save_flg:
        if len(files_name)==2:
            if feature=='PST12':
                np.save('X_'+files_name[0]+'_p='+str(persistence)+'.npy', Feature1)
                np.save('X_'+files_name[1]+'_p='+str(persistence)+'.npy', Feature2)
            else:
                np.save('X_'+files_name[0]+'.npy', Feature1)
                np.save('X_'+files_name[1]+'.npy', Feature2)
        elif len(files_name)==1:
            np.save('X_'+files_name[0]+'.npy', Feature1)
    else:
        return Feature1,Feature2

def main(feature_list,PDBid,Chain,mutations,num_sites,pH,type_list,max_mut,site_method,persistence=0):
    get_structure(PDBid, Chain,mutations,num_sites, pH,type_list)
    pro_wt=pro_complex(PDBid, Chain, mutations,'WT')
    pro_mt=pro_complex(PDBid, Chain, mutations,'MT')
    for feature in feature_list:
        save_feature(pro_wt,pro_mt,feature,site_method=site_method,max_mut=max_mut,persistence=persistence)
def main_NMR(feature_list,PDBid,Chain,mutations,num_sites,pH,type_list,max_mut,site_method,MODEL_ID,persistence=0):
    for feature in feature_list:
        Feature1=[]
        Feature2=[]

        for model_id in MODEL_ID:
            pdbid=PDBid+'_m'+str(model_id)
            get_structure(pdbid, Chain,mutations,num_sites, pH,type_list)
            pro_wt=pro_complex(pdbid, Chain, mutations,'WT')
            pro_mt=pro_complex(pdbid, Chain, mutations,'MT')
            a,b=save_feature(pro_wt,pro_mt,feature,site_method=site_method,max_mut=max_mut,save_flg=False,persistence=persistence)
            Feature1.append(a)
            Feature2.append(b)
        files_name = FEATURE_FILE_NAME.get(feature)
        if len(files_name) == 2:
            np.save('X_' + files_name[0] + '.npy', Feature1)
            np.save('X_' + files_name[1] + '.npy', Feature2)
        elif len(files_name) == 1:
            np.save('X_' + files_name[0] + '.npy', Feature1)

if __name__ == '__main__':
    PDBid='1nd4'
    mutations=['RA18A']
    Chain='A'
    num_sites=1
    pH='7.0'
    type_list=['WT','MT']
    # main(PDBid,Chain,mutations,num_sites,pH,type_list)


