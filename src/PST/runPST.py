import numpy as np
from .PST_features import *
from .PST_structure import *
def get_features(pro_mt,site_method,max_mut):
    """""
        get features for one genotype 
    """""
    mute0_dth, mute0_bar, mute12_a, mute12_all = PH_mute(pro_mt)
    print('PH done')
    mute0_betti0, mute0_nonzero=PST_mute(pro_mt)
    print('PST done')

    PH0_mutation = merge_features_site(mute0_bar,site_method=site_method,max_mut=max_mut)
    PH0dth_mutation = merge_features_site(mute0_dth,site_method=site_method,max_mut=max_mut)
    PH12_mutation = merge_features_site(mute12_a, mute12_all,site_method=site_method,max_mut=max_mut)
    PST_mutation_PH0=merge_features_site(mute0_betti0,site_method=site_method,max_mut=max_mut)
    PST_mutation = merge_features_site(mute0_nonzero,site_method=site_method,max_mut=max_mut)
    return PH0_mutation,PH0dth_mutation,PH12_mutation,PST_mutation_PH0,PST_mutation
def save_feature(pro_wt,pro_mt,site_method,max_mut=1,save_flg=True):
    PH0_mt, PH0dth_mt, PH12_mt, PSTPH0_mt, PST_mt=get_features(pro_mt,site_method=site_method,max_mut=max_mut)
    PH0_wt, PH0dth_wt, PH12_wt, PSTPH0_wt, PST_wt=get_features(pro_wt,site_method=site_method,max_mut=max_mut)



    PH0_mutation = construct_features_MT_WT(PH0_mt, PH0_wt)
    PH0dth_mutation = construct_features_MT_WT(PH0dth_mt, PH0dth_wt)
    PH12_mutation = construct_features_MT_WT(PH12_mt,PH12_wt)
    PST_mutation_PH0 = construct_features_MT_WT(PSTPH0_mt,PSTPH0_wt)
    PST_mutation = construct_features_MT_WT(PST_mt,PST_wt)
    PST=np.concatenate((PST_mutation_PH0,PH12_mutation))
    PST=np.concatenate((PST,PST_mutation))
    PH=np.concatenate((PH0_mutation,PH12_mutation))
    if save_flg:
        assert len(PH) > 0
        # 'PSTmutePH0 PH12mute PSTmute'
        # 'PH0mute PH12mute'
        np.save('X_PH.npy',PH)
        np.save('X_PST.npy',PST)
        # np.save('X_PH12mute.npy',PH12_mutation)
        # np.save('X_PSTmutePH0.npy',PST_mutation_PH0)
        # np.save('X_PSTmute.npy',PST_mutation)
    #return PH0_mutation,PH0dth_mutation,PH12_mutation,PST_mutation_PH0,PST_mutation
    return PH,PST
def main(PDBid,Chain,mutations,num_sites,pH,type_list,max_mut,site_method):
    get_structure(PDBid, Chain,mutations,num_sites, pH,type_list)
    pro_wt=pro_complex(PDBid, Chain, mutations,'WT')
    pro_mt=pro_complex(PDBid, Chain, mutations,'MT')
    save_feature(pro_wt,pro_mt,site_method=site_method,max_mut=max_mut)
def main_NMR(PDBid,Chain,mutations,num_sites,pH,type_list,max_mut,site_method,MODEL_ID):
    # PH0_mutation=[]
    # PH0dth_mutation=[]
    # PH12_mutation=[]
    # PST_mutation_PH0=[]
    # PST_mutation=[]
    PH=[]
    PST=[]
    for model_id in MODEL_ID:
        pdbid=PDBid+'_m'+str(model_id)
        get_structure(pdbid, Chain,mutations,num_sites, pH,type_list)
        pro_wt=pro_complex(pdbid, Chain, mutations,'WT')
        pro_mt=pro_complex(pdbid, Chain, mutations,'MT')
        a,b=save_feature(pro_wt,pro_mt,site_method=site_method,max_mut=max_mut,save_flg=False)
        PH.append(a)
        PST.append(b)
    assert len(PST)>0
    # np.save('X_PH0mute.npy',PH0_mutation)
    # np.save('X_PH0dthmute.npy',PH0dth_mutation)
    # np.save('X_PH12mute.npy',PH12_mutation)
    # np.save('X_PSTmutePH0.npy',PST_mutation_PH0)
    # np.save('X_PSTmute.npy',PST_mutation)
    np.save('X_PH.npy', PH)
    np.save('X_PST.npy', PST)
if __name__ == '__main__':
    PDBid='1nd4'
    mutations=['RA18A']
    Chain='A'
    num_sites=1
    pH='7.0'
    type_list=['WT','MT']
    # main(PDBid,Chain,mutations,num_sites,pH,type_list)


