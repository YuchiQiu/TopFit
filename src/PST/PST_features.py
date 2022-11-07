import gudhi, re, os, sys
import gudhi.representations
import numpy as np
from  math  import sqrt
radii = {'C':0.77,'N':0.70,'O':0.73,'S':1.03,'P':1.10,'F':0.72,'CL':0.99,'BR':1.14,'I':1.33,'H':0.37}

FRIDefault = [['Lorentz', 0.5,  5],
              ['Lorentz', 1.0,  5],
              ['Lorentz', 2.0,  5],
              ['Exp',     1.0, 15],
              ['Exp',     2.0, 15]]
ElementList = ['C', 'N', 'O']
ElementTau = np.array([6., 1.12, 1.1])
EleLength = len(ElementList)
dt = np.dtype([('dim', int), ('birth', float), ('death', float)])

class atom:
    def __init__(self, position, chain, resID, atype):
        self.pos   = np.array(position)
        self.chain = chain
        self.resID = resID
        self.atype = atype

class pro_complex:
    # def __init__(self, PDBid, Chains, muteChain, resName, resID):
    def __init__(self, PDBid, Chain, mutations,genotype):
        self.PDBid = PDBid
        # self.Antibody = list(Antibody)
        # self.Antigen  = list(Antigen)
        self.Chain   = Chain
        self.genotype=genotype
        self.num_site   = len(mutations)
        self.muteList={}
        for mut in mutations:
            self.muteList[mut[1]] = []
        for mut in mutations:
            self.muteList[mut[1]].append([mut[0],mut[len(mut)-1],int(mut[2:len(mut)-1])])
        # point cloud of atoms 
        # '_b': binding;                 '_m': mutant
        # '_m': includes mutant residue; '_o': includes other part
        #                                '_s': includes other side
        self.atoms_m_m={}
        for muteChain in self.muteList:
            self.atoms_m_m[muteChain] = []
        self.atoms_m_o={}
        for muteChain in self.muteList:
            self.atoms_m_o[muteChain] = []
        for muteChain in self.muteList:
            for mute_info in self.muteList[muteChain]:
                # resWT = mute_info[0]
                resID = mute_info[2]
                PDBfile=self.PDBid+'_'+self.genotype+'_'+muteChain+str(resID)+'.pdb'
                atoms_m_m,atoms_m_o = loadMutantPDB(PDBfile,muteChain,resID)
                self.atoms_m_m[muteChain].append(atoms_m_m)
                self.atoms_m_o[muteChain].append(atoms_m_o)


#### load PDB files

def loadBindingPDB(PDBfile,Antibody,Antigen):
    """""
    Get atoms near binding interface

    Parameters
    ----------
    PDBfile: str
        PDB file name for atoms near mutational site
    Antibody: str
        chains for antibody
    Antigen: str
        chains for antigen

    Returns
    -------
    atoms_b_1: list
        atoms at antibody
        each element stores a list of `atom` with the same element type (given in @ElementList)
    atoms_b_2: list
        atoms at antigen
        each element stores a list of `atom` with the same element type (given in @ElementList)
    """
    # read pdb data for binding interface.
    # '_1': antibody; '_2': antigen
    atoms_b_1 = [[] for _ in range(EleLength)]
    atoms_b_2 = [[] for _ in range(EleLength)]
    # PDBfile = self.PDBid+'_'+''.join(Antibody)+'_'+''.join(Antigen)+'_'+genotype
    fp = open(PDBfile)
    for line in fp:
        if line[:4]=='ATOM':
            resID = int(line[23:26])
            if line[21] in Antibody and line[13] in ElementList:
                Atom = atom([float(line[30:38]), float(line[38:46]), float(line[46:54])],
                            line[21], resID, line[13])
                atoms_b_1[ElementList.index(Atom.atype)].append(Atom)
            elif line[21] in Antigen and line[13] in ElementList:
                Atom = atom([float(line[30:38]), float(line[38:46]), float(line[46:54])],
                            line[21], resID, line[13])
                atoms_b_2[ElementList.index(Atom.atype)].append(Atom)
    fp.close()

    return atoms_b_1,atoms_b_2
def loadMutantPDB(PDBfile,muteChain,resID):
    """""
    Get atoms at one mutational site

    Parameters
    ----------
    PDBfile: str
        PDB file name for atoms near mutational site
    muteChain: str (one letter)
        chain for the mutation
    resID: int
        residue ID for the mutational site

    Returns
    -------
    atoms_m_m: list
        atoms at mutational site
        each element stores a list of `atom` with the same element type (given in @ElementList)
    atoms_m_m: list
        atoms near mutational site
        each element stores a list of `atom` with the same element type (given in @ElementList)
    """
    atoms_m_m = [[] for _ in range(EleLength)]
    atoms_m_o = [[] for _ in range(EleLength)]

    # PDBfile = self.PDBid+'_'+''.join(self.Antibody)+'_'+''.join(self.Antigen)+'_'+genotype+'_'+muteChain+str(resID)
    fp = open(PDBfile)
    for line in fp:
        if line[:4]=='ATOM':
            if line[21]==muteChain and line[13] in ElementList and int(line[23:26])==resID:
                Atom = atom([float(line[30:38]), float(line[38:46]), float(line[46:54])],
                            line[21], int(line[23:26]), line[13])
                atoms_m_m[ElementList.index(Atom.atype)].append(Atom)
            elif line[13] in ElementList:
                Atom = atom([float(line[30:38]), float(line[38:46]), float(line[46:54])],
                            line[21], int(line[23:26]), line[13])
                atoms_m_o[ElementList.index(Atom.atype)].append(Atom)

    fp.close()
    # self.atoms_m_m[muteChain].append(atoms_m_m)
    # self.atoms_m_o[muteChain].append(atoms_m_o)
    return atoms_m_m,atoms_m_o


#### PH features

def rips_complex(atoms_m, atoms_o, interval=1., birth_cut=2., death_cut=11.):
    """""
    0-d PH features using rips complex.
    By considering pairwise and element-wise interactions between two sets of atoms
    Bipartite distance is used between two sets of point cloud

    Parameters
    ----------
    atoms_m: list
        each element stores a list of `atom` with the same element type (given in @ElementList)
    atoms_o: list
        each element stores a list of `atom` with the same element type (given in @ElementList)
    interval: float
    birth_cut: float
    death_cut: float
        Rips features are extracted in the interval [`birth_cut`, `death_cut`] for filtration parameter
        This interval is cut into several bins with size `interval`
    Returns
    -------
    rips_dth: 2D numpy array (#bins X #element^2)
    rips_bar: 2D numpy array (#bins X #element^2)
        #element^2 groups of atoms are constructed
        by considering pairwise interactions between atoms
        with different elements for `atoms_m` and `atoms_b`
        In each bin: `rips_dth` calculates number of persistent bars die in the bin.
                     `rips_bar` calculates number of persistent bars exist in the bin.
    """
    PHcut = death_cut+1.
    BinIdx = int((death_cut-birth_cut)/interval)
    Bins = np.linspace(birth_cut, death_cut, BinIdx+1)
    def BinID(x):
        for i in range(BinIdx):
            if Bins[i] <= x < Bins[i+1]:
                return i
        return BinIdx

    rips_dth = np.zeros([BinIdx, EleLength*EleLength])
    rips_bar = np.zeros([BinIdx, EleLength*EleLength])

    for idx_m, e_m in enumerate(ElementList):
        for idx_o, e_o in enumerate(ElementList):
            length_m = len(atoms_m[idx_m])
            length_o = len(atoms_o[idx_o])
            matrixA = np.ones((length_m+length_o, length_m+length_o))*100.
            for ii, iatom in enumerate(atoms_m[idx_m]):
                for jj, jatom in enumerate(atoms_o[idx_o]):
                    dis = np.linalg.norm(iatom.pos-jatom.pos)
                    matrixA[ii, length_m+jj] = dis
                    matrixA[length_m+jj, ii] = dis
            rips_complex = gudhi.RipsComplex(distance_matrix=matrixA, max_edge_length=PHcut)
            PH = rips_complex.create_simplex_tree().persistence()

            tmpbars = np.zeros(length_m+length_o, dtype=dt)
            cnt = 0
            for simplex in PH:
                dim, b, d = int(simplex[0]), float(simplex[1][0]), float(simplex[1][1])
                if d-b < 0.1: continue
                tmpbars[cnt]['dim']   = dim
                tmpbars[cnt]['birth'] = b
                tmpbars[cnt]['death'] = d
                cnt += 1
            bars = tmpbars[0:cnt]
            for bar in bars:
                death = bar['death']
                if death>=death_cut or death<birth_cut: continue
                Did = BinID(death)
                rips_dth[ Did, idx_m*EleLength+idx_o] += 1
                rips_bar[:Did, idx_m*EleLength+idx_o] += 1
    return np.array(rips_dth), np.array(rips_bar)

def alpha_shape(atoms_m, atoms_o):
    """""
        1-d and 2-d PH features using alpha complex.
        Euclidean distance is used by concatenate all atoms.

        Parameters
        ----------
        atoms_m: list
            each element stores a list of `atom` with the same element type (given in @ElementList)
        atoms_o: list
            each element stores a list of `atom` with the same element type (given in @ElementList)
        Returns
        -------
        alpha_PH12: 3D numpy array (#element X #element X 14)
            (#element X #element) groups of interactions between different types of elements.
            14=2X7:
                2: two types of PH featuers: 1D and 2D
                7=3+2+2:
                    3 types of statistical values for lengths of all betti bars:
                        sum, max, mean
                    2 types of statistics for birth values of all betti bars:
                        max, min
                    2 types of statistics for death values of all betti bars:
                        max, min
        alpha_PH12_all: 1D numpy array (dimension 14)
            14=2X7 values for betti bars for atoms without classfied their elements.
        """
    alpha_PH12 = np.zeros([EleLength, EleLength, 14])
    for idx_m, e_m in enumerate(ElementList):
        for idx_o, e_o in enumerate(ElementList):
            points  = [iatom.pos for iatom in atoms_m[idx_m]]
            points += [iatom.pos for iatom in atoms_o[idx_o]]
            alpha_complex = gudhi.AlphaComplex(points=points)
            PH = alpha_complex.create_simplex_tree().persistence()

            tmpbars = np.zeros(len(PH), dtype=dt)
            cnt = 0
            for simplex in PH:
                dim, b, d = int(simplex[0]), float(simplex[1][0]), float(simplex[1][1])
                if d-b < 0.1: continue
                tmpbars[cnt]['dim']   = dim
                tmpbars[cnt]['birth'] = b
                tmpbars[cnt]['death'] = d
                cnt += 1
            bars = tmpbars[0:cnt]
            if len(bars[bars['dim']==1]['death']) > 0:
                alpha_PH12[idx_m, idx_o, 0] = np.sum(bars[bars['dim']==1]['death'] - \
                                                     bars[bars['dim']==1]['birth'])
                alpha_PH12[idx_m, idx_o, 1] = np.max(bars[bars['dim']==1]['death'] - \
                                                     bars[bars['dim']==1]['birth'])
                alpha_PH12[idx_m, idx_o, 2] = np.mean(bars[bars['dim']==1]['death'] - \
                                                      bars[bars['dim']==1]['birth'])
                alpha_PH12[idx_m, idx_o, 3] = np.min(bars[bars['dim']==1]['birth'])
                alpha_PH12[idx_m, idx_o, 4] = np.max(bars[bars['dim']==1]['birth'])
                alpha_PH12[idx_m, idx_o, 5] = np.min(bars[bars['dim']==1]['death'])
                alpha_PH12[idx_m, idx_o, 6] = np.max(bars[bars['dim']==1]['death'])
            if len(bars[bars['dim']==2]['death']) > 0:
                alpha_PH12[idx_m, idx_o, 7]  = np.sum(bars[bars['dim']==2]['death'] - \
                                                      bars[bars['dim']==2]['birth'])
                alpha_PH12[idx_m, idx_o, 8]  = np.max(bars[bars['dim']==2]['death'] - \
                                                      bars[bars['dim']==2]['birth'])
                alpha_PH12[idx_m, idx_o, 9]  = np.mean(bars[bars['dim']==2]['death'] - \
                                                       bars[bars['dim']==2]['birth'])
                alpha_PH12[idx_m, idx_o, 10] = np.min(bars[bars['dim']==2]['birth'])
                alpha_PH12[idx_m, idx_o, 11] = np.max(bars[bars['dim']==2]['birth'])
                alpha_PH12[idx_m, idx_o, 12] = np.min(bars[bars['dim']==2]['death'])
                alpha_PH12[idx_m, idx_o, 13] = np.max(bars[bars['dim']==2]['death'])

    alpha_PH12_all = np.zeros([14])
    points = []
    for idx in range(EleLength):
        points += [iatom.pos for iatom in atoms_m[idx]]
        points += [iatom.pos for iatom in atoms_o[idx]]
    alpha_complex = gudhi.AlphaComplex(points=points)
    PH = alpha_complex.create_simplex_tree().persistence()

    tmpbars = np.zeros(len(PH), dtype=dt)
    cnt = 0
    for simplex in PH:
        dim, b, d = int(simplex[0]), float(simplex[1][0]), float(simplex[1][1])
        if d-b < 0.1: continue
        tmpbars[cnt]['dim']   = dim
        tmpbars[cnt]['birth'] = b
        tmpbars[cnt]['death'] = d
        cnt += 1
    bars = tmpbars[0:cnt]
    if len(bars[bars['dim']==1]['death']) > 0:
        alpha_PH12_all[0] = np.sum(bars[bars['dim']==1]['death'] - \
                                   bars[bars['dim']==1]['birth'])
        alpha_PH12_all[1] = np.max(bars[bars['dim']==1]['death'] - \
                                   bars[bars['dim']==1]['birth'])
        alpha_PH12_all[2] = np.mean(bars[bars['dim']==1]['death'] - \
                                    bars[bars['dim']==1]['birth'])
        alpha_PH12_all[3] = np.min(bars[bars['dim']==1]['birth'])
        alpha_PH12_all[4] = np.max(bars[bars['dim']==1]['birth'])
        alpha_PH12_all[5] = np.min(bars[bars['dim']==1]['death'])
        alpha_PH12_all[6] = np.max(bars[bars['dim']==1]['death'])
    if len(bars[bars['dim']==2]['death']) > 0:
        alpha_PH12_all[7]  = np.sum(bars[bars['dim']==2]['death'] - \
                                    bars[bars['dim']==2]['birth'])
        alpha_PH12_all[8]  = np.max(bars[bars['dim']==2]['death'] - \
                                    bars[bars['dim']==2]['birth'])
        alpha_PH12_all[9]  = np.mean(bars[bars['dim']==2]['death'] - \
                                     bars[bars['dim']==2]['birth'])
        alpha_PH12_all[10] = np.min(bars[bars['dim']==2]['birth'])
        alpha_PH12_all[11] = np.max(bars[bars['dim']==2]['birth'])
        alpha_PH12_all[12] = np.min(bars[bars['dim']==2]['death'])
        alpha_PH12_all[13] = np.max(bars[bars['dim']==2]['death'])

    return np.array(alpha_PH12), np.array(alpha_PH12_all)
def alpha_shape_landscape(atoms_m, atoms_o,genotype,
                          resolution=10,num_landscapes=3):
    """""
        1-d and 2-d PH features using alpha complex.
        Landscape representation is used.
        Euclidean distance is used by concatenate all atoms.

        Parameters
        ----------
        atoms_m: list
            each element stores a list of `atom` with the same element type (given in @ElementList)
        atoms_o: list
            each element stores a list of `atom` with the same element type (given in @ElementList)
        Returns
        """
    # acX = gd.AlphaComplex(points=X).create_simplex_tree()
    # dgmX = acX.persistence()
    # LS = gd.representations.Landscape(resolution=1000)
    # L = LS.fit_transform([acX.persistence_intervals_in_dimension(1)])
    # import matplotlib.pyplot as plt

    alpha_PH12=[]
    for idx_m, e_m in enumerate(ElementList):
        for idx_o, e_o in enumerate(ElementList):
            points  = [iatom.pos for iatom in atoms_m[idx_m]]
            points += [iatom.pos for iatom in atoms_o[idx_o]]
            alpha_complex = gudhi.AlphaComplex(points=points)
            acX = alpha_complex.create_simplex_tree()
            dgmY=acX.persistence()
            LS = gudhi.representations.Landscape(resolution=resolution,num_landscapes=num_landscapes)
            L1 = LS.fit_transform([acX.persistence_intervals_in_dimension(1)])
            alpha_PH12.extend(L1[0])
            L2 = LS.fit_transform([acX.persistence_intervals_in_dimension(2)])
            alpha_PH12.extend(L2[0])

            # gudhi.plot_persistence_diagram(dgmY)
            # plt.savefig('PD_' + genotype + '_'+e_m+e_o+'.pdf')
            # plt.close()
            # gudhi.plot_persistence_barcode(dgmY)
            # plt.savefig('barcode_' + genotype + '_' + e_m + e_o + '.pdf')
            # plt.close()
            # for i in range(num_landscapes):
            #     plt.plot(L1[0][resolution * (i - 1):resolution * i])
            # plt.savefig('landscape_d=1' + genotype + '_'+e_m+e_o+ '.pdf')
            # plt.close()
            # for i in range(num_landscapes):
            #     plt.plot(L2[0][resolution * (i - 1):resolution * i])
            # plt.savefig('landscape_d=2' + genotype + '_'+e_m+e_o+ '.pdf')
            # np.save('point_cloud_' + genotype + '_'+e_m+e_o+ '.npy', points)

    alpha_PH12_all=[]
    points = []
    for idx in range(EleLength):
        points += [iatom.pos for iatom in atoms_m[idx]]
        points += [iatom.pos for iatom in atoms_o[idx]]
    # np.save('point_cloud_'+genotype+'.npy',points)
    alpha_complex = gudhi.AlphaComplex(points=points)
    acX = alpha_complex.create_simplex_tree()#.persistence()
    dgmY=acX.persistence()
    LS = gudhi.representations.Landscape(resolution=resolution, num_landscapes=num_landscapes)
    L1 = LS.fit_transform([acX.persistence_intervals_in_dimension(1)])

    # gudhi.plot_persistence_diagram(dgmY)
    # plt.savefig('PD_'+genotype+'.pdf')
    # plt.close()
    # gudhi.plot_persistence_barcode(dgmY)
    # plt.savefig('barcode_' + genotype+'.pdf')
    # plt.close()

    alpha_PH12_all.extend(L1[0])
    L2 = LS.fit_transform([acX.persistence_intervals_in_dimension(2)])
    alpha_PH12_all.extend(L2[0])

    # for i in range(num_landscapes):
    #     plt.plot(L1[0][resolution*(i-1):resolution*i])
    # plt.savefig('landscape_d=1'+genotype+'.pdf')
    # plt.close()
    # for i in range(num_landscapes):
    #     plt.plot(L2[0][resolution*(i-1):resolution*i])
    # plt.savefig('landscape_d=2'+genotype+'.pdf')
    # plt.close()
    alpha_PH12=np.array(alpha_PH12)
    alpha_PH12_all=np.array(alpha_PH12_all)
    print(alpha_PH12.shape)
    print(alpha_PH12_all.shape)
    return alpha_PH12,alpha_PH12_all
def PH_mute0(pro):
    """""
        PH0 features for `pro` at mutational sites

        Parameters
        ----------
        pro: pro_complex

        Returns
        -------
        0-d PH features:
            mute0_dth: list, dimension: number of mutational sites
            mute0_bar: list, dimension: number of mutational sites
        """
    mute0_dth = []
    mute0_bar= []
    # mute12_a = []
    # mute12_all= []

    for muteChain in pro.muteList:
        for i in range(len(pro.muteList[muteChain])):
            tmp1,tmp2 = rips_complex(pro.atoms_m_m[muteChain][i],pro.atoms_m_o[muteChain][i])
            # tmp3,tmp4 = alpha_shape(pro.atoms_m_m[muteChain][i],pro.atoms_m_o[muteChain][i])
            mute0_dth.append(np.asarray(tmp1))
            mute0_bar.append(np.asarray(tmp2))
            # mute12_a.append(np.asarray(tmp3))
            # mute12_all.append(np.asarray(tmp4))
    mute0_dth = np.asarray(mute0_dth)
    mute0_bar = np.asarray(mute0_bar)
    # mute12_a = np.asarray(mute12_a)
    # mute12_all = np.asarray(mute12_all)
    return mute0_dth,mute0_bar #,mute12_a,mute12_all

def PH_mute12(pro):
    """""
        PH12 features for `pro` at mutational sites

        Parameters
        ----------
        pro: pro_complex

        Returns
        -------
        1-d and 2-d PH features:
            mute12_a: list, dimension: number of mutational sites
            mute12_all: list, dimension: number of mutational sites
        """
    # mute0_dth = []
    # mute0_bar= []
    mute12_a = []
    mute12_all= []

    for muteChain in pro.muteList:
        for i in range(len(pro.muteList[muteChain])):
            # tmp1,tmp2 = rips_complex(pro.atoms_m_m[muteChain][i],pro.atoms_m_o[muteChain][i])
            tmp3,tmp4 = alpha_shape(pro.atoms_m_m[muteChain][i],pro.atoms_m_o[muteChain][i])
            # mute0_dth.append(np.asarray(tmp1))
            # mute0_bar.append(np.asarray(tmp2))
            mute12_a.append(np.asarray(tmp3))
            mute12_all.append(np.asarray(tmp4))
    # mute0_dth = np.asarray(mute0_dth)
    # mute0_bar = np.asarray(mute0_bar)
    mute12_a = np.asarray(mute12_a)
    mute12_all = np.asarray(mute12_all)
    # return mute0_dth,mute0_bar,mute12_a,mute12_all
    return mute12_a,mute12_all

def PH_mute_landscape(pro):
    """""
        PH features for `pro` at mutational sites

        Parameters
        ----------
        pro: pro_complex

        Returns
        -------
        0-d PH features:
            mute0_dth: list, dimension: number of mutational sites
            mute0_bar: list, dimension: number of mutational sites
        1-d and 2-d PH features using persistence landscape
            mute12_a: list, dimension: number of mutational sites
            mute12_all: list, dimension: number of mutational sites
        """
    # binding0_dth, binding0_bar = rips_complex(pro.atoms_b_1, pro.atoms_b_2)
    # binding12_a, binding12_all = alpha_shape(pro.atoms_b_1, pro.atoms_b_2)
    # mute0_dth={}
    # mute0_bar={}
    # mute12_a={}
    # mute12_all={}
    # mute0_dth = []
    # mute0_bar= []
    mute12_a = []
    mute12_all= []

    for muteChain in pro.muteList:
        for i in range(len(pro.muteList[muteChain])):
            # tmp1,tmp2 = rips_complex(pro.atoms_m_m[muteChain][i],pro.atoms_m_o[muteChain][i])
            tmp3,tmp4 = alpha_shape_landscape(pro.atoms_m_m[muteChain][i],pro.atoms_m_o[muteChain][i],pro.genotype)
            # mute0_dth.append(np.asarray(tmp1))
            # mute0_bar.append(np.asarray(tmp2))
            mute12_a.append(np.asarray(tmp3))
            mute12_all.append(np.asarray(tmp4))
    # mute0_dth = np.asarray(mute0_dth)
    # mute0_bar = np.asarray(mute0_bar)
    mute12_a = np.asarray(mute12_a)
    mute12_all = np.asarray(mute12_all)
    return mute12_a,mute12_all

#### PST features

def L0_rips(atoms_m,atoms_o,cut,VDW=False,birth_cut=2., death_cut=11.):
    # Bipartite graph between atoms_m and atoms_o is constructed
    # cut: filtration parameters for constructing simplicial complex
    # calculate non-zero spectra of 0-combinatorial Laplacian matrix
    # VDW: options for the interaction distance. Two atoms interactions are omitted if their distance is below birth_cut
    #      If consider interactions between binding interface, take constant default birth_cut=2
    #      If consider interactions between residue, take birth_cut as the sum of van der waals radii of two atoms.
    PHcut=12.0
    betti0=[]
    nonzero=[]
    for idx_m, e_m in enumerate(ElementList):
        for idx_o, e_o in enumerate(ElementList):
            if VDW:
                ele_m=ElementList[idx_m]
                ele_o=ElementList[idx_o]
                birth_cut = radii.get(ele_m)+radii.get(ele_o)
            length_m = len(atoms_m[idx_m])
            length_o = len(atoms_o[idx_o])
            # matrixA = np.zeros((length_m+length_o, length_m+length_o))
            matrixA = np.ones((length_m + length_o, length_m + length_o)) * 100.
            Laplacian = np.zeros((length_m + length_o, length_m + length_o))

            for ii, iatom in enumerate(atoms_m[idx_m]):
                for jj, jatom in enumerate(atoms_o[idx_o]):
                    dis = np.linalg.norm(iatom.pos-jatom.pos)
                    if dis>=birth_cut:
                        matrixA[ii, length_m+jj] = dis
                        matrixA[length_m+jj, ii] = dis
                    if dis<=cut and dis>=birth_cut:
                        Laplacian[ii, length_m+jj] = -1
                        Laplacian[length_m+jj, ii] = -1
            Diagonal=np.diagflat(-np.sum(Laplacian,axis=0))
            Laplacian=Laplacian+Diagonal
            LAMBDA = np.linalg.eigvalsh(Laplacian)
            rips_complex = gudhi.RipsComplex(distance_matrix=matrixA, max_edge_length=PHcut)
            PH = rips_complex.create_simplex_tree().persistence()

            #b00: betti0 for the entire graph
            b00=0
            # b0: betti0 by excluding short bars (<birth_cut) (to exclude covalent interactions)
            # and long bars (>death_cut) (to exclude bars never dead in the cutoff distance).
            b0=0
            for simplex in PH:
                dim, b, d = int(simplex[0]), float(simplex[1][0]), float(simplex[1][1])
                if d>cut and b<=cut and d >= birth_cut:
                    b00+=1
                # else:
                #     print(simplex)
                if d >= cut and b <= cut and d <= death_cut and d >= birth_cut:
                    b0+=1
            betti0.append(b0)
            assert b00==np.sum(LAMBDA<10**-10); 'betti 0 not equal to number of zero eigenvalues'
            eigens=LAMBDA[LAMBDA>10**-10]
            if len(eigens)>0:
                lll=[np.min(eigens),np.max(eigens),np.mean(eigens),np.std(eigens),np.sum(eigens)]
                nonzero.append(np.asarray(lll))
            else:
                lll=[0.,0.,0.,0.,0.]
                nonzero.append(np.asarray(lll))
    betti0=np.asarray(betti0)
    nonzero=np.asarray(nonzero)
    return betti0,nonzero

def PST_L0(atoms_m,atoms_o,interval = 1., birth_cut = 2., death_cut = 11.):

    betti0 = []
    nonzero = []

    BinIdx = int((death_cut - birth_cut) / interval)
    Bins = np.linspace(birth_cut, death_cut, BinIdx + 1)
    for cut in Bins:
        a,b = L0_rips(atoms_m, atoms_o, cut,VDW=True)
        betti0.extend(a)
        nonzero.extend(b)
    return np.asarray(betti0),np.asarray(nonzero)
def PST_mute(pro):
    """""
        PST L0 features for `pro` at mutational sites

        Parameters
        ----------
        pro: pro_complex

        Returns
        -------
        """
    mute0_betti0=[]
    mute0_nonzero=[]
    for muteChain in pro.muteList:
        for i in range(len(pro.muteList[muteChain])):
            a,b = PST_L0(pro.atoms_m_m[muteChain][i],pro.atoms_m_o[muteChain][i])
            mute0_betti0.append(a)
            mute0_nonzero.append(b)
    mute0_betti0=np.asarray(mute0_betti0)
    mute0_nonzero=np.asarray(mute0_nonzero)
    return mute0_betti0,mute0_nonzero
def PST_L1L2_mute(pro,persistence=0,num=100000000,interval = 1., birth_cut = 2., death_cut = 11.):
    """""
        PST features for `pro` at mutational sites

        Parameters
        ----------
        pro: pro_complex

        persistence: persistence parameter

        num: number of eigenvalues to be evaluated. default make it very large to contain all eigenvalues
        Returns
        -------
        1-d and 2-d non-harmonic spectra:
            mute12_a: list, dimension: number of mutational sites
            mute12_all: list, dimension: number of mutational sites
        """
    from filepath_dir import software_path
    path = software_path()
    HERMES_path = path['HERMES_path']
    os.system('cp ' + HERMES_path + 'Snapshot .')

    BinIdx = int((death_cut - birth_cut) / interval)
    Bins = np.linspace(birth_cut, death_cut, BinIdx + 1)
    filtration_txt='filtration.txt'
    file = open(filtration_txt, 'w')
    for p in Bins:
        file.write(str(p**2)+' ')
    file.write('\n')
    file.close()
    alpha_PST12_betti=[]
    alpha_PST12_nonzero=[]
    for muteChain in pro.muteList:
        for i in range(len(pro.muteList[muteChain])):
            tmp1,tmp2=PST_L1L2_spectra(pro.atoms_m_m[muteChain][i],pro.atoms_m_o[muteChain][i],
                             filtration_txt, persistence, num)
            alpha_PST12_betti.append(tmp1)
            alpha_PST12_nonzero.append(tmp2)
    os.system('rm Snapshot')
    os.system('rm *.vtk')
    os.system('rm filtration.txt')
    os.system('rm sorted_alpha.txt')
    os.system('rm points.xyz')
    alpha_PST12_betti=np.asarray(alpha_PST12_betti)
    alpha_PST12_nonzero=np.asarray(alpha_PST12_nonzero)
    return alpha_PST12_betti,alpha_PST12_nonzero
def PST_L1L2_spectra(atoms_m, atoms_o,filtration_txt,persistence,num):
    """""
        Non-harmonic spectra for L1 and L2, generated by HERMES.
        Euclidean distance is used by concatenate all atoms.

        Parameters
        ----------
        atoms_m: list
            each element stores a list of `atom` with the same element type (given in @ElementList)
        atoms_o: list
            each element stores a list of `atom` with the same element type (given in @ElementList)
        Returns
        -------

        """

    alpha_PST12_nonzero=[]#np.zeros([EleLength, EleLength, len(Bins),10])
    alpha_PST12_betti=[]
    for idx_m, e_m in enumerate(ElementList):
        for idx_o, e_o in enumerate(ElementList):
            points  = [iatom.pos for iatom in atoms_m[idx_m]]
            points += [iatom.pos for iatom in atoms_o[idx_o]]
            L0,L1,L2=run_hermes(points,filtration_txt,num,persistence)
            a,b=stat_eigen(L1)
            alpha_PST12_betti.extend(a)
            alpha_PST12_nonzero.extend(b)
            a,b=stat_eigen(L2)
            alpha_PST12_betti.extend(a)
            alpha_PST12_nonzero.extend(b)
    # alpha_PST12_all_nonzero=[]
    # alpha_PST12_all_betti=[]

    points = []
    for idx in range(EleLength):
        points += [iatom.pos for iatom in atoms_m[idx]]
        points += [iatom.pos for iatom in atoms_o[idx]]
    L0, L1, L2 = run_hermes(points, filtration_txt, num, persistence)
    a, b = stat_eigen(L1)
    alpha_PST12_betti.extend(a)
    alpha_PST12_nonzero.extend(b)
    a, b = stat_eigen(L2)
    alpha_PST12_betti.extend(a)
    alpha_PST12_nonzero.extend(b)
        # xyz_file=genereate_xyz(points)
        # os.system('./Snapshot ' + xyz_file + ' ' + filtration_txt + ' ' + str(num) + ' ' + str(persistence))
    # print(alpha_PST12)
    # print(alpha_PST12_all)
    # print(len(alpha_PST12))
    # print(len(alpha_PST12_all))
    # hhh
    alpha_PST12_betti = np.array(alpha_PST12_betti)
    alpha_PST12_nonzero = np.array(alpha_PST12_nonzero)


    return alpha_PST12_betti,alpha_PST12_nonzero


def read_spectra(filename):
    L = []
    tmp = open(filename, 'r').readlines()
    for spectra in tmp:
        if len(spectra.replace('\n', '')) == 0:
            L.append(np.array([]))
        else:
            spectra = spectra.replace('\n', '').split()
            spectra = np.array([float(a) for a in spectra])
            spectra[spectra < 10 ** -10] = 0.
            L.append(spectra)
    return L
def run_hermes(points,filtration_txt,num,persistence):
    """""
    generate a .xyz files for point cloud
    
    """""
    xyz_file='points.xyz'
    file = open(xyz_file, 'w')
    for pts in points:
        file.write(str(pts[0])+'   '+str(pts[1])+'   '+str(pts[2])+'\n')
    file.close()
    os.system('./Snapshot ' + xyz_file + ' ' + filtration_txt + ' ' + str(num) + ' ' + str(persistence))
    L0=read_spectra('snapshots_vertex.txt')
    L1=read_spectra('snapshots_edge.txt')
    L2=read_spectra('snapshots_facet.txt')
    os.system('rm snapshots_vertex.txt')
    os.system('rm snapshots_edge.txt')
    os.system('rm snapshots_facet.txt')

    return L0,L1,L2

    # return xyz_file
def stat_eigen(L):
    # only look at non-harmonic eigenvalues
    nonzero=[]
    betti=[]
    for l in L:
        betti.append((l==0.0).sum())
        l=l[l>0.]
        if len(l) > 0:
            lll = [np.min(l), np.max(l), np.mean(l), np.std(l), np.sum(l)]
            nonzero.extend(lll)
        else:
            lll = [0., 0., 0., 0., 0.]
            nonzero.extend(lll)
    nonzero=np.array(nonzero)
    return betti, nonzero
####   construct features


def merge_features(*args):
    Features=np.array([])
    for x in args:
        Features=np.concatenate((Features,x.flatten()))
    # print(Features.shape)
    return Features
def merge_features_site(*args,site_method,max_mut=1):
    Features=np.array([])
    for x in args:
        # print(x.shape)
        if site_method=='sum':
            Features=np.concatenate((Features,np.sum(x,axis=0).flatten()))
        elif site_method=='avg':
            Features=np.concatenate((Features,np.mean(x,axis=0).flatten()))
        elif site_method=='align':
            l=x.shape[0]
            pad_l=max_mut-l
            feature=x.reshape(-1)
            pad_size=np.prod(list(x.shape[1:]))
            assert len(feature) == pad_size*l
            if pad_l>0:
                feature=np.append(feature,np.zeros(pad_size*pad_l))
            Features=np.concatenate((Features,feature))

    # print(Features.shape)
    return Features

def construct_features_MT_WT(f_MT,f_WT):
    if f_MT is not None:
        # print(f_MT)
        # print(f_WT)
        Feature = np.concatenate((f_MT.flatten(),f_WT.flatten()))
        Feature = np.concatenate((Feature.flatten(),f_MT.flatten()-f_WT.flatten()))
        return Feature
    else:
        return None