"""
Discriminating GH7 CBHs and EGs with position-specific classification rules
"""




# Imports
#================#
import pandas as pd
import numpy as np
from scipy import stats

from Bio.PDB.PDBParser import PDBParser
from Bio.PDB import Selection
import matplotlib.pyplot as plt

import bioinformatics as bioinf

import warnings
warnings.filterwarnings("ignore")






# Prepare sequences and data
#=================================#
if __name__ == '__main__':
    # Get MSA with only TreCel7A positions for analysis
    heads, sequences = bioinf.split_fasta('fasta/structure_based_alignment/' \
                                          '/cel7_nr99_structaln.fasta')
    trecel7a_seq = sequences[0]
    trecel7a_positions = [x for x in range(len(trecel7a_seq)) 
                            if trecel7a_seq[x].isalpha()]
    sequences_treonly = []
    for i in range(len(sequences)):
        seq = list(sequences[i])
        seq = [seq[x] for x in trecel7a_positions]
        seq = ''.join(seq)
        sequences_treonly.append(seq)
    bioinf.combine_fasta(heads, sequences_treonly, 'fasta/trecel7a_positions_only/' \
                         'cel7_all.fasta')
    
    
    # Separate sequences in MSA to two sub-MSAs (CBH and EG)
    subtype = list(pd.read_csv('results_final/cel7_subtypes.csv')['ncbi_pred_class'])
    cbh_pos = [x for x in range(len(subtype)) if subtype[x]==1]
    egl_pos = [x for x in range(len(subtype)) if subtype[x]==0]
    heads_cbh = [heads[x] for x in cbh_pos]
    
    sequences_cbh = [sequences_treonly[x] for x in cbh_pos]
    bioinf.combine_fasta(heads_cbh, sequences_cbh, 'fasta/trecel7a_positions_only/' \
                         'cbh_all.fasta')
    
    heads_egl = [heads[x] for x in egl_pos]
    sequences_egl = [sequences_treonly[x] for x in egl_pos]
    bioinf.combine_fasta(heads_egl, sequences_egl, 'fasta/trecel7a_positions_only/' \
                         'egl_all.fasta')
    
    # Save MSA of only catalytic domain
    sequences_cat = [seq[17:451] for seq in sequences_treonly]
    bioinf.combine_fasta(heads, sequences_cat, 'fasta/trecel7a_positions_only/' \
                         'cel7_cat.fasta')
    
    seq_cat_cbh = [sequences_cat[x] for x in cbh_pos]
    bioinf.combine_fasta(heads_cbh, seq_cat_cbh, 'fasta/trecel7a_positions_only/' \
                         'cbh_cat.fasta')
    
    seq_cat_egl = [sequences_cat[x] for x in egl_pos]
    bioinf.combine_fasta(heads_egl, seq_cat_egl, 'fasta/trecel7a_positions_only/' \
                         'egl_cat.fasta')






# Create a Class for efficient analysis of MSA
#================================================#

class Cel7MSA():
    '''A class for efficient analyses of GH7 MSA, and for 
        deriving position-specific classification rules. 
        cbh_msa is the fasta file of the subalignment containing 
        CBH sequences and only TreCel7A positions. egl_msa if the
        fasta file for EG sequences.'''
        
        
    def __init__(self, cbh_msa, egl_msa):
        self.cbh_msa = cbh_msa
        self.egl_msa = egl_msa
        self._cbh_color = 'blue'
        self._egl_color = 'red'
        self.cbh_size = len(bioinf.split_fasta(self.cbh_msa)[1])
        self.egl_size = len(bioinf.split_fasta(self.egl_msa)[1])
        
        
    def _get_aa_freq(self, fasta, analysis='amino', include_gaps=True):
        '''Return a dataframe of the frequencies of all 20 amino acids (AAs)
        in each site of an MSA. If include_gaps=True, gaps are treated as
        AAs and are included in the analysis. 
        If analysis == 'amino', frequencies of AAs are computed.
        if analysis == 'type', frequencies of AA types are computed'''
        
        if analysis=='amino':
            fasta_df = bioinf.fasta_to_df(fasta)
            amino_acids = list('ACDEFGHIKLMNPQRSTVWY')
        elif analysis=='type':
            # Replace AA single letter with single letter describing
            # the AA type
            # Aliphatic (A), Aromatic (R), Polar (P), Positve (T), 
            # and Negative (N)
            fasta_df = bioinf.residue_to_group(fasta)
            amino_acids = list('ARPTN') 
        if include_gaps:
            amino_acids += ['-']
            
        # Determine frequency
        store = []
        length = len(fasta_df.index)
        for k in range(len(fasta_df.columns)):
            aa_list = list(fasta_df.iloc[:,k])
            aa_count = [aa_list.count(x)/length for x in amino_acids]
            store.append(aa_count)
        store = pd.DataFrame(store).transpose()
        store.index = amino_acids
        return store
        
    
    def get_freq(self, analysis='amino', include_gaps=True):
        '''Determine the amino acid frequencies of positions
        in CBH and EG subalignments.'''
        
        self.cbh_freq = self._get_aa_freq(self.cbh_msa, analysis='amino', 
                                          include_gaps=include_gaps)
        self.egl_freq = self._get_aa_freq(self.egl_msa, analysis='amino', 
                                          include_gaps=include_gaps)

            
    def get_consensus_sequences(self):
        '''Determine the consensus sequence for CBH and EG from
        the MSAs.'''
        
        cbh_cons, egl_cons = '', '' # Initialize empty string
        amino_acids = list(self.cbh_freq.index)
        
        # Loop through each position and determine the most frequent amino acid
        for i in range(len(self.cbh_freq.columns)):
            c_freq = list(self.cbh_freq.iloc[:,i])
            e_freq = list(self.egl_freq.iloc[:,i])
            cbh_cons += amino_acids[c_freq.index(max(c_freq))]
            egl_cons += amino_acids[e_freq.index(max(e_freq))]
        self.consensus_cbh = cbh_cons
        self.consensus_egl = egl_cons
    
    def _one_to_three(self, one):
        '''Convert one-letter amino acid to three'''
        ones = list('ACDEFGHIKLMNPQRSTVWY')
        threes = ['Ala', 'Cys', 'Asp', 'Glu', 'Phe', 'Gly', 'His', 'Ile', 'Lys', 'Leu',
                  'Met', 'Asn', 'Pro', 'Gln', 'Arg', 'Ser', 'Thr', 'Val', 'Trp', 'Tyr']
        return threes[ones.index(one)]
    
    
    def get_rules(self, analysis='amino'):
        '''Derive position-specific classification rules for discriminating 
        Cel7 CBHs from EGs using the consensus residue (or residue type) 
        from the MSA.'''
        
        [cbh_freq, egl_freq] = [self._get_aa_freq(x, analysis=analysis, include_gaps=True) 
                                for x in [self.cbh_msa, self.egl_msa]]
        if analysis=='type':
            ind = ['ALI', 'ARO', 'POL', 'POS', 'NEG', '-']
            cbh_freq.index, egl_freq.index = ind, ind
            
        # Empty lists for storing results
        pos, sens, spec, acc, mcc, rule, pvalue = [], [], [], [], [], [], []
        
        # Loop through each position, derive rules, and test the rules
        for i in range(len(cbh_freq.columns)):
            cbh_cons_freq = cbh_freq.iloc[:,i].max() # Frequency of consensus AA/type
            cbh_cons_type = cbh_freq.index[list(cbh_freq.iloc[:,i]).index(cbh_cons_freq)]
            egl_cons_freq = egl_freq.iloc[:,i].max()
            egl_cons_type = egl_freq.index[list(egl_freq.iloc[:,i]).index(egl_cons_freq)]
            
            # Rule 1: [X ==> CBH, not X ==> EGL]
            if cbh_cons_type != '-':
                sensitivity = cbh_cons_freq   # X => CBH
                cons_pos = list(cbh_freq.index).index(cbh_cons_type)
                specificity = 1 - egl_freq.iloc[cons_pos,i] # not X => EGL
                accuracy = (sensitivity*self.cbh_size + specificity*self.egl_size)/(self.cbh_size + self.egl_size)
                tp = sensitivity * self.cbh_size  # X => CBH
                fp = self.egl_size*egl_freq.iloc[cons_pos,i] # X => EGL
                tn = specificity*self.egl_size  # not X => EG
                fn = self.cbh_size * (1 - cbh_freq.iloc[cons_pos,i]) # not X => CBH
                MCC = ((tp * tn) - (fp * fn))/np.sqrt((tp + fp)*(tp + fn)*(tn + fp)*(tn + fn))
                table = np.array([[tp, fp], [fn, tn]]) # CBH and EG have same contingency table
                p_value = stats.chi2_contingency(table)[1]
                
                pos.append(i+1) # Position in TreCel7A
                sens.append(sensitivity * 100)
                spec.append(specificity * 100)
                acc.append(accuracy * 100)
                mcc.append(MCC)
                pvalue.append(p_value)
                key = self._one_to_three(cbh_cons_type) if analysis=='amino' else cbh_cons_type
                rule.append(f'{key}=>CBH, not {key}=>EGL')
            
            
            # Rule 2: [Z ==> EGL, not Z ==> CBH]
            if egl_cons_type != '-':
                cons_pos = list(egl_freq.index).index(egl_cons_type)
                specificity = egl_cons_freq  # Z => EGL
                sensitivity = 1 - cbh_freq.iloc[cons_pos,i]  # not Z => CBH
                accuracy = (sensitivity*self.cbh_size + specificity*self.egl_size)/(self.cbh_size + self.egl_size)
                tp = self.cbh_size * sensitivity # not Z => CBH
                fp = self.egl_size * (1 - egl_freq.iloc[cons_pos,i]) # not Z => EGL
                tn = self.egl_size * specificity # Z => EGL
                fn = self.cbh_size * cbh_freq.iloc[cons_pos,i] # Z => CBH
                MCC = ((tp * tn) - (fp * fn))/np.sqrt((tp + fp)*(tp + fn)*(tn + fp)*(tn + fn))
                table = np.array([[tp, fp], [fn, tn]])
                p_value = stats.chi2_contingency(table)[1]
                
                pos.append(i+1)
                sens.append(sensitivity * 100)
                spec.append(specificity * 100)
                acc.append(accuracy * 100)
                mcc.append(MCC)
                pvalue.append(p_value)
                key = self._one_to_three(egl_cons_type) if analysis=='amino' else egl_cons_type
                rule.append(f'not {key}=>CBH, {key}=>EGL')
            
            # Rule 3: [X ==> CBH, Z ==> EGL]
            if cbh_cons_type != egl_cons_type and '-' not in [cbh_cons_type, egl_cons_type]:
                #sensitivity, specificity = cbh_cons_freq, egl_cons_freq
                cons_posX = list(cbh_freq.index).index(cbh_cons_type)
                cons_posZ = list(egl_freq.index).index(egl_cons_type)
                #accuracy = (sensitivity*self.cbh_size + specificity*self.egl_size)/(self.cbh_size + self.egl_size)
                tpX = cbh_cons_freq * self.cbh_size # X => CBH
                fpX = egl_freq.iloc[cons_posX,i] * self.egl_size  # X => EGL
                tnX = (1 - egl_freq.iloc[cons_posX,i]) * self.egl_size  # not X => EG
                fnX = self.cbh_size * (1 - cbh_freq.iloc[cons_posX,i]) # not X => CBH
                
                tpZ = self.cbh_size * (1 - cbh_freq.iloc[cons_posZ,i]) # not Z => CBH
                fpZ = self.egl_size * (1 - egl_freq.iloc[cons_posZ,i]) # not Z => EGL
                tnZ = egl_cons_freq * self.egl_size # Y => EGL
                fnZ = cbh_freq.iloc[cons_posZ,i] * self.cbh_size # Y => CBH
                
                tp, fp, tn, fn = tpX + tpZ, fpX + fpZ, tnX + tnZ, fnX + fnZ
                MCC = ((tp * tn) - (fp * fn))/np.sqrt((tp + fp)*(tp + fn)*(tn + fp)*(tn + fn))
                cbh_table = np.array([[tp, fp], [fn, tn]])
                p_value = stats.chi2_contingency(cbh_table)[1]
                
                pos.append(i+1)
                sens.append(tp/(tp + fn) * 100)
                spec.append(tn/(tn + fp) * 100)
                acc.append((tn + tp)/(tn + tp + fp + fn) * 100)
                mcc.append(MCC)
                pvalue.append(p_value)
                keyX = self._one_to_three(cbh_cons_type) if analysis=='amino' else cbh_cons_type
                keyZ = self._one_to_three(egl_cons_type) if analysis=='amino' else egl_cons_type
                rule.append(f'{keyX}=>CBH, {keyZ}=>EGL')
                
        store = pd.DataFrame([pos, rule, sens, spec, acc, mcc, pvalue]).transpose()
        store.columns = ['tre_pos', 'rule', 'sens', 'spec', 'acc', 'mcc', 'pvalue']
        return store
            
    
    def site_plot(self, site, savefig=False, savepath=None):
        '''Plot bar graphs of amino acid composition for site.'''
        cbh_comp = self.cbh_freq.iloc[:20,site-1]*100
        egl_comp = self.egl_freq.iloc[:20,site-1]*100
        
        lw = 1.0   # Width of bar edge
        w = 0.25   #  Width of bar
        fnt = 'Arial'
        ticks_font = {'fontname':fnt, 'size':'20'}
        label_font = {'family':fnt, 'size':'22'}
        title_font = {'family':fnt, 'size':'24'}
        legend_font = {'family':'Arial', 'size':'18'}
        legend_label = ['CBH', 'EG']
        plt.rcParams['grid.alpha'] = 0.5
        
        X = np.arange(len(cbh_comp))
        
        out_cbh = plt.bar(X-0.5*w, cbh_comp, color='blue', width=w, linewidth=lw,
                       edgecolor='black')
        out_egl = plt.bar(X+0.5*w, egl_comp, color='red', width=w, linewidth=lw,
                       edgecolor='black')
        
        plt.yticks(**ticks_font)
        plt.xticks(X, cbh_comp.index, rotation=0, **ticks_font)
        #plt.grid(True, linestyle='--')
        plt.ylabel('Frequency (%)', **label_font)
        plt.title(f'POS{site}', **title_font)
        pltout = [x[0] for x in [out_cbh, out_egl]]
        plt.legend(pltout, legend_label, frameon=1, numpoints=1, shadow=1, loc='middle top', 
                   prop=legend_font)
        plt.tight_layout()
        
        if savefig:
           plt.savefig(f'{savepath}/pos{site}.pdf')
        plt.show()






# Distance between each residue and all glycosyl subsites
#============================================================#
if __name__ == '__main__':
	# Get pdb data (4C4C)
	structure_id = '4c4c'
	filename = 'fasta/4c4c.pdb'
	parser=PDBParser(PERMISSIVE=1)
	structure = parser.get_structure(structure_id,filename)
	model = structure[0]
	chain = model['A']
	reslist = chain.get_list()


	def distance(x,y):
			'''Return the euclidean distance between 2 3D vectors.'''
			return np.sqrt((x[0]-y[0])**2 + (x[1]-y[1])**2 + (x[2]-y[2])**2)
 

	def atom_distance(x,y):
		'''Returns the closest distance between two objects (x and y), i.e. the distance
		between the closest atoms in x and y.'''
		[x_atoms, y_atoms] = [Selection.unfold_entities(obj, 'A') for obj in [x,y]]
		distances = []
		for xatom in x_atoms:
			for yatom in y_atoms:
				distances.append(distance(xatom.get_coord(), yatom.get_coord()))
		return min(distances)


	reslist = Selection.unfold_entities(structure,'R')  # all residues
	ligand_ids = [('H_BGC', x, ' ') for x in range(435,444)] # ids for 9 glycosyl residues in BGC
	ligand_res = [chain[x] for x in ligand_ids]    # residue objects of 9 glycosyl residues in BGC
	prot_res = reslist[:434]   # protein residues


	# Calculate distances between closest atoms of protein residues and glycosyl residues
	store = []
	for prot in prot_res:
		p_store = []
		for lig in ligand_res:
			p_store.append(atom_distance(prot,lig))
		store.append(p_store)
	store = pd.DataFrame(store)
	store.index = np.array(store.index)+1
	store.columns = ['+2', '+1', '-1', '-2', '-3', '-4', '-5', '-6', '-7']


	# Closest subsite
	min_dist = []
	closest_res = []
	for i in range(len(store.index)):
		distances = list(store.iloc[i,:])
		min_dist.append(min(distances))
		closest_res.append(store.columns[distances.index(min(distances))])
	dist_store = pd.DataFrame([closest_res, min_dist]).transpose()
	dist_store.index = store.index
	dist_store.columns = ['closest_subsite', 'distance']
	dist_store.to_csv('results_final/residue_distances.csv')






# Derive classification rules using the Cel7MSA class
#=========================================================#
if __name__ == '__main__':
    cbhmsa = 'fasta/trecel7a_positions_only/cbh_cat.fasta'
    eglmsa = 'fasta/trecel7a_positions_only/egl_cat.fasta'
    cel7msa = Cel7MSA(cbhmsa, eglmsa)
    cel7msa.get_freq(include_gaps=True)
    rules_amino = cel7msa.get_rules(analysis='amino')
    rules_amino['closest_subsite'] = [dist_store[0][x] for x in rules_amino.tre_pos]
    rules_amino['dist_subsite'] = [dist_store[1][x] for x in rules_amino.tre_pos]
    rules_type = cel7msa.get_rules(analysis='type')
    rules_type['closest_subsite'] = [dist_store[0][x] for x in rules_type.tre_pos]
    rules_type['dist_subsite'] = [dist_store[1][x] for x in rules_type.tre_pos]
    rules = rules_amino.append(rules_type, ignore_index=True)
    
    # Save rules
    rules_amino.to_csv('results_final/rules/rules_amino.csv')
    rules_type.to_csv('results_final/rules/rules_type.csv')
    rules.to_csv('results_final/rules/rules_all.csv')
    
    
    # Get consensus sequences
    cel7msa.get_consensus_sequences()
    consensus_cbh = cel7msa.consensus_cbh
    consensus_egl = cel7msa.consensus_egl
    bioinf.combine_fasta(['CBH consensus', 'EG consensus'],[consensus_cbh, consensus_egl],
                         'fasta/trecel7a_positions_only/consensus.fasta')



