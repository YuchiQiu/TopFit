import os, re, sys, warnings
from Bio.PDB import PDBParser
from Bio.SeqIO.FastaIO import SimpleFastaParser
from Bio.PDB.Polypeptide import three_to_one
import numpy as np
warnings.filterwarnings('ignore')

AminoA = ['ARG', 'HIS', 'LYS', 'ASP', 'GLU', 'SER', 'THR', 'ASN', 'GLN', 'CYS',
          'SEF', 'GLY', 'PRO', 'ALA', 'VAL', 'ILE', 'LEU', 'MET', 'PHE', 'TYR',
          'TRP']
default_cutoff = '13.'


class get_structure:
    # def __init__(self, PDBid, Antibody, Antigen, muteChain, resWT, resID, resMT, pH, cutoff=default_cutoff):
    def __init__(self, PDBid, Chain, mutations,num_sites, pH,type_list,cutoff=default_cutoff):

        from filepath_dir import software_path
        path=software_path()
        self.jackal_path=path['jackal_path']
        self.vmd_path=path['vmd_path']
        # self.pqr_path=path['pqr_path']
        self.PDBid    = PDBid
        self.Chain   = Chain
        self.muteList={}
        for mut in mutations:
            self.muteList[mut[1]] = []
        for mut in mutations:
            self.muteList[mut[1]].append([mut[0],mut[len(mut)-1],int(mut[2:len(mut)-1])])
        self.fasta    = {}
        self.cutoff   = cutoff

        # filename
        # self.fileComplex   = PDBid+'_'+Antibody+'_'+Antigen
        # self.fileAntibody  = PDBid+'_'+Antibody
        # self.fileAntigen   = PDBid+'_'+Antigen
        # check if the PDB has the wildtype residue given in self.muteList
        # use Biopython to load PDB_file
        parser = PDBParser(PERMISSIVE=1)
        s = parser.get_structure(PDBid, PDBid+'_WT.pdb')
        for muteChain in self.muteList:
            for iresidue in s[0][muteChain]:
                for mute_info in self.muteList[muteChain]:
                    resWT = mute_info[0]
                    resID = mute_info[2]
                    if iresidue.id==(' ',resID,' '):
                        if resWT!=three_to_one(iresidue.resname):
                            sys.exit('Wrong residue name for input!!!')

        os.system('cp ' + self.jackal_path + 'profix .')
        os.system('cp ' + self.jackal_path + 'scap .')
        os.system('cp ' + self.jackal_path + 'jackal.dir .')
        # generate mutant PDB
        filename = self.PDBid
        if not os.path.exists(filename+'_MT.pdb'):
            if num_sites==0:
                os.system('mv '+filename+'_WT.pdb '+filename+'_MT.pdb')
            else:
                scap_file = open('tmp_scap.list', 'w')
                for muteChain in self.muteList:
                    for mute_info in self.muteList[muteChain]:
                        resWT = mute_info[0]
                        resMT = mute_info[1]
                        resID = mute_info[2]
                        if resWT!=resMT:
                            scap_file.write(','.join([muteChain, str(resID), resMT])+'\n')
                scap_file.close()
                filename=self.PDBid
                os.system('./scap -ini 20 -min 4 '+filename+'_WT.pdb tmp_scap.list')
                # os.system('mv '+filename+'_WT_scap.pdb '+filename+'_MT.pdb')
                os.system('./scap -ini 20 -min 4 '+filename+'_WT_scap.pdb')
                os.system('mv '+filename+'_WT_scap_scap.pdb '+filename+'_MT.pdb')
                os.system('rm tmp_scap.list')
        os.system('rm scap')
        os.system('rm profix')
        os.system('rm jackal.dir')

        # self.get_fasta()


        # generate cutoff PDB for atoms near the binding sites '_b.pdb'
        # if not os.path.exists(self.fileComplex + '_WT_b.pdb'):
        #     self.select_pdb_binding('WT')
        # for genotype in type_list:
        #     if not os.path.exists(self.fileComplex + '_'+genotype+'_b.pdb'):
        #         self.select_pdb_binding(genotype)

        # generate cutoff PDB_file
        for muteChain in self.muteList:
            for mute_info in self.muteList[muteChain]:
                # resWT = mute_info[0]
                # resMT = mute_info[1]
                resID = mute_info[2]
                # if not os.path.exists(self.fileComplex + '_WT_'+muteChain+str(resID)+'.pdb'):
                #     self.select_pdb_mute('WT', muteChain, resID)
                for genotype in type_list:
                    if not os.path.exists(self.PDBid + '_'+genotype+'_' + muteChain + str(resID) + '.pdb'):
                        self.select_pdb_mute(genotype, muteChain, resID)


        # write up fasta_file only for mutant protein

    def select_chains(self,inputfile,chainList,file_chain,genotype,VMD=False):
        # Two methods for selecting chains. VMD and loop. Some magic bugs exist.
        # The combination uses VMD first and loop later can avoid erros in PDB files?
        filename = self.PDBid + '_' + file_chain+ '_'+genotype+'.pdb'
        runID=False
        if not os.path.exists(filename):
            # print(filename)
            runID=True
            if not VMD:
                lines=open(inputfile).readlines()
                output=open(filename,'w')
                for line in lines:
                    if line[0:4]=='ATOM' and line[21] in chainList and line[17:20] in AminoA:
                        output.write(line)
                    elif line[0:3]=='TER':
                        output.write('TER\n')
                output.close()
            else:
                vmdtcl = open('vmd.tcl', 'w')
                vmdtcl.write('mol new {' + inputfile +'} type {pdb} first 0 last 0 step 1 waitfor 1\n')
                vmdtcl.write('set prot1 [atomselect top "protein and chain ' + ' '.join(chainList) + '"]\n')
                vmdtcl.write('$prot1 writepdb ' + filename + '\n')
                vmdtcl.write('exit')
                vmdtcl.close()
                os.system(self.vmd_path + 'vmd -dispdev text -e vmd.tcl')
                os.system('rm vmd.tcl')
        return filename,runID
    def select_pdb_mute(self,genotype,muteChain,resID):
        vmdtcl = open('vmd.tcl', 'w')
        vmdtcl.write('mol new {' + self.PDBid +'_'+genotype+ '.pdb} type {pdb} first 0 last 0 step 1 waitfor 1\n')
        vmdtcl.write('set prot1 [atomselect top "within ' + self.cutoff +
                     ' of resid ' + str(resID) + ' and chain ' + muteChain + '"]\n')
        vmdtcl.write('$prot1 writepdb ' + self.PDBid + '_'+genotype+'_'+muteChain+str(resID)+\
                     '.pdb\n')
        vmdtcl.write('exit')
        vmdtcl.close()
        os.system(self.vmd_path + 'vmd -dispdev text -e vmd.tcl')
        os.system('rm vmd.tcl')


    def clean_files(self):
        os.system('rm -rf *.pdb')
        os.system('rm scap')
        os.system('rm jackal.dir')
        os.system('rm profix')

if __name__ == "__main__":
    PDBid='1nd4'
    mutations=['RA18A']
    Chain='A'
    num_sites=1
    pH='7.0'
    type_list=['WT','MT']
    get_structure(PDBid, Chain,mutations,num_sites, pH,type_list)