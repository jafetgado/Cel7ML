"""
Adhoc functions for bioinformatic analyses
"""


import pandas as pd
import numpy as np

from Bio.Align.Applications import MafftCommandline


#=======================================================#
def deliner(path):
	'''Remove line breaks within sequence data of a fasta
    file (path) and save the output by overwritting the 
    original file.'''

	file_input = open(path, 'r')
	data = []
	
	for line in file_input:
		text = line.strip()

		if text.startswith('>'):
			data.append(text)
			is_header = True
		else:
			if is_header == True:
				data.append(text)						
			elif is_header == False:
				data[-1] += text
			is_header = False
	
	file_input.close()

	with open(path, 'w') as file_output:
		for ele in data:
			file_output.write(ele + '\n')


#=======================================================#
def seq_identity(seq1, seq2, gap = '-', short=True):
    ''' Calculate the percentage sequence identity between
    two aligned sequences relative to the shortest 
    sequence. If short is set to false, the identity is 
    calculated relative to the longest sequence.'''

    s1 = np.fromstring(seq1, dtype="S1")
    s2 = np.fromstring(seq2, dtype="S1")
    num1 = np.sum(s1!=b"-")
    num2 = np.sum(s2!=b"-")
    iden = np.sum(np.logical_and(s1!=b"-" , s1==s2))
    
    if short:
        return iden/float(min(num1, num2)) * 100
    else:
        return iden/float(max(num1,num2)) * 100


#=======================================================#
def split_fasta(path):
	'''Read a fasta file and return a list of two lists,
    [[headers], [sequences]], with the same order as the
    fasta file.'''
    
	fileobj = open(path, 'r')    
	headers, seqdata = [],[]

	for line in fileobj:
		if line.startswith('>'):
			headers.append(line.strip().replace('>', ''))
		else:
			seqdata.append(line.strip())
	fileobj.close()	
	return [headers, seqdata]


#=======================================================#
def combine_fasta(headers, seqdata, path):
	'''Write a fasta file (path) from a list of headers 
    and sequences, [[headers], [seqdata]], with the same 
    order as the lists.'''
    
	with open(path, 'w') as pp:
		for i in range(len(headers)):
			pp.write('>' + headers[i] + '\n' + seqdata[i] + '\n')


#=======================================================#
def Mafft_MSA(path1, path2, Mafft_exe='/usr/local/bin/mafft'):
    '''Implement MSA with MAFFT on sequences in fasta file 
    (path1) and save aligned sequences in path2. Specify the
    the path of the MAFFT executable with Mafft_exe.'''    
    
    deliner(path1)
    [h1, s1] = split_fasta(path1)
    Mafft_cline = MafftCommandline(Mafft_exe, input=path1)
    stdout, stderr = Mafft_cline()
    
    with open(path2, 'w') as store:
        store.write(stdout)
    deliner(path2)
    [h2, s2] = split_fasta(path2)
    
    with open(path2, 'w') as fileobj:
        for i in range(len(h1)):
            posi = h2.index(h1[i])
            fileobj.write('>' + h2[posi] + '\n' + s2[posi] + '\n')


#=======================================================#
def get_accession(path):
    ''' Return a list of all accession codes from a 
    fasta file (path) with the header in Genbank format.'''
    
    [h,s] = split_fasta(path)
    return [x.split()[0] for x in h]
    

#=======================================================#
def fasta_to_df(fasta):
    ''' Read an MSA in fasta format and return a dataFrame
    whose indexes are the sequences and columns are the 
    sites in the MSA.'''
    
    deliner(fasta)   # Sequence data in fasta file should be a single line
    [h,s] = split_fasta(fasta)
    data = []
    for i in range(len(s)):
        data.append(list(s[i]))    # Add each sequence as a list of residues/gaps
    df = pd.DataFrame(data)
    df.index = [x.replace('>', '') for x in h]
    return df


#=======================================================#
def df_to_fasta(df, fasta):
    ''' Write an MSA dataFrame as a fasta file.'''
    
    h =[str(x) for x in df.index]
    s = [''.join(list(df.iloc[x,:])) for x in range(len(df))]
    combine_fasta(h,s,fasta)


#=======================================================#
def trim_gaps_df(df, max_perc, gap='-', verbose=True):
    ''' Remove all sites in an MSA DataFrame which have 
    gaps in more than or equal to max_perc percent of 
    the sequences.'''
	
    size = len(df)
    max_gap = max_perc * size/100.0
    col_del = []
    
    for i in range(len(df.columns)):
        if list(df.iloc[:,i]).count(gap) >= max_gap:
            col_del.append(i)
    newdf = df.drop(df.columns[col_del], axis=1)
    newdf.columns = range(len(newdf.columns))
    if verbose:
        print('\nDeleted ' + str(len(col_del)) + ' sites\n')	
    return newdf


    
#=======================================================#
def  residue_to_group(fasta):
    '''Convert the 20 amino acids in an MSA (fasta) to the
    following letters denoting the stereochemical group of
    each amino acid. Return result as a dataframe.
    
    A - Aliphatic (A, G, V, L, I, M, C),
    R - Aromatic (F, W, Y, H)
    P - Polar (S, T, N, Q)
    T - Positive (K, R)
    N - Negative (D, E) '''
    
    store = {'A':'A', 'G':'A', 'V':'A', 'L':'A', 'I':'A', 'M':'A', 'C':'A', 'P':'A',\
             'F':'R', 'W':'R', 'Y':'R', 'H':'R',\
             'S':'P', 'T':'P', 'N':'P', 'Q':'P',\
             'K':'T', 'R':'T',\
             'D':'N', 'E':'N',\
             '-':'-', 'X':'-', 'Z':'-'}
    
    df = fasta_to_df(fasta)
    newdf = pd.DataFrame()
    
    for i in range(len(df.columns)):
        for j in range(len(df.index)):
            aa = df.iloc[j,i]
            newdf.set_value(index=j, col=i, value=store[aa])
    
    newdf.index = df.index
    return newdf 


#=======================================================#
def msa_to_resid(seq, msapos):
    '''For a sequence (seq) in a MSA (i.e. with gaps) and
    a site in the MSA (msapos), return the position in 
    the sequence without gaps (i.e. resid).'''
    
    seq_nogap = seq.replace('-','')
    before, after = seq[:msapos], seq[msapos:]
    before_nogap, after_nogap = before.replace('-',''), after.replace('-','')
    if seq_nogap.index(after_nogap)==len(before_nogap):
        return seq_nogap.index(after_nogap)
    else:
        return len(seq_nogap) - 1


#=======================================================#
def resid_to_msa(seq,resid):
    '''For a sequence (seq) in a MSA and the position in 
    the sequence without gaps (resid), return  the
    corresponding position (site) in the MSA.'''
    
    seqlist = list(seq)
    num = 0
    for i in range(len(seqlist)):
        if seqlist[i].isalpha():
            seqlist[i] = num
            num += 1
    try:
        return seqlist.index(resid)
    except:
        return 'Not Found\n'


#=======================================================#