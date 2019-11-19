#========================================#
# Analyze results and plot figures
#========================================#




# Imports
#==============#

import pandas as pd
import numpy as np
import scipy
import random

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

import bioinformatics as bioinf
   



# Plots for HMM method 5-fold cross validation 
#===============================================#
fnt='Arial'
ticks_font = {'fontname':fnt, 'size':'18'}
legend_font = {'family':fnt, 'size':'12'}
label_font = {'family':fnt, 'size':'20'}
plt.rcParams['figure.figsize'] = [6,3]

ec = 'black'
legend_label = ['CBH', 'EG']

# SwissProt Dataset
ex = pd.read_excel('results_final/swiss_kfold.xlsx')
lw = 0.8

out1 = plt.bar(range(30), ex.diff_score[:30], color='blue', 
               linewidth=lw, edgecolor=ec)
out2 = plt.bar(range(30,44), ex.diff_score[30:], color='red', 
               linewidth=lw, edgecolor=ec)
pltout = [x[0] for x in [out1, out2]]

plt.xlabel('Sequence', **label_font)
plt.ylabel('Score difference', **label_font)
plt.xticks(**ticks_font)
plt.yticks([-300,-150,0,150,300,450], **ticks_font)
plt.xlim([-0.6,43.6])
plt.axhline(color='black', linewidth=1)
plt.legend(pltout, legend_label, prop=legend_font, 
           loc='upper right')
plt.tight_layout()
plt.savefig('plots/swiss_kfold.pdf')
plt.close()

# NCBI dataset
ex = pd.read_excel('results_final/ncbi_kfold.xlsx')
lw = 0.15
cbhs = list(ex.diff_score[:291])
egs = list(ex.diff_score[291:])
random.shuffle(cbhs)
random.shuffle(egs)
out1 = plt.bar(range(291), cbhs, color='blue', linewidth=lw, 
               edgecolor='blue')
out2 = plt.bar(range(291,427), egs, color='red', linewidth=lw, 
               edgecolor='red')
pltout = [x[0] for x in [out1, out2]]

plt.xlabel('Sequence', **label_font)
plt.ylabel('Score difference', **label_font)
plt.xticks(**ticks_font)
plt.yticks([-300,-150,0,150,300,450], **ticks_font)
plt.xlim([-1,428])
plt.axhline(color='black', linewidth=1)
plt.legend(pltout, legend_label, prop=legend_font, 
           loc='upper right')
plt.tight_layout()
plt.savefig('plots/ncbi_kfold.pdf')
plt.close()




# Loop positions in TreCel7A and TreCel7B for pymol
#====================================================#

# Cel7A
loopstart = [98, 399, 369, 383, 51, 194, 244, 339]
length = [5,13,5,10,6,8,10,4]
cel7a_start = list(loopstart)
cel7a_stop = [loopstart[i] + length[i] - 1 for i in range(8)]
cel7a_pymol = 'select cel7a_loops, '
for i in range(8):
    cel7a_pymol += 'resi {0}-{1} or '.format(
                        cel7a_start[i], cel7a_stop[i])

# Cel7B
fasta = 'fasta/structure_based_alignment/structure6_mafft.fasta'
heads, seqs = bioinf.split_fasta(fasta)
seq7a_msa, seq7b_msa = seqs[0], seqs[3]
seq7a, seq7b = seq7a_msa.replace('-', ''), seq7b_msa.replace('-','')
msastart = [bioinf.resid_to_msa(seq7a_msa, x-1) for x in cel7a_start]
msastop = [bioinf.resid_to_msa(seq7a_msa, x-1) for x in cel7a_stop]
cel7b_start = [bioinf.msa_to_resid(seq7b_msa, x) for x in msastart]
cel7b_stop = [bioinf.msa_to_resid(seq7b_msa, x+1) for x in msastop]
cel7b_pymol = 'select cel7b_loops, '
for i in range(8):
    cel7b_pymol += 'resi {0}-{1} or '.format(
                        cel7b_start[i] + 1, cel7b_stop[i])

# Write
with open('plots/loops_pymol.txt', 'w') as pymol:
    pymol.write(cel7a_pymol[:-4] + '\n\n')
    pymol.write(cel7b_pymol[:-4])




# ML subtype performance table
#==============================================#
mlkeys = ['dec', 'svm', 'knn', 'log']
features = ['A1', 'A2', 'A3', 'A4', 'B1', 'B2', 'B3', 'B4', 'all8']

store2 = []
columns = []
for key in mlkeys:
    excel = pd.read_excel('results_final/ml_subtype_pred/{0}.xlsx'.format(key),
                          index_col=0)
    sens_store, spec_store, acc_store = [], [], []
    columns.extend([key + '_sens', key + '_spec', key + '_acc'])
    for i in range(len(features)):
        sens_store.append(str(round(excel.sens_mean[i], 1)) + ' ± ' + \
                      str(round(excel.sens_std[i], 1)))
        spec_store.append(str(round(excel.spec_mean[i], 1)) + ' ± ' + \
                      str(round(excel.spec_std[i], 1)))
        acc_store.append(str(round(excel.acc_mean[i], 1)) + ' ± ' + \
                      str(round(excel.acc_std[i], 1)))
        
    store2.extend([sens_store, spec_store, acc_store])
store2 = pd.DataFrame(store2).transpose()
store2.index = features
store2.columns = columns
store2.to_excel('plots/ml_subtype_table.xlsx')




# ML subtype MCC plot
#=======================#
mlkeys = ['dec', 'log', 'knn', 'svm']
features = ['A1', 'A2', 'A3', 'A4', 'B1', 'B2', 'B3', 'B4', 'all8']

store2 = []
columns = []
for key in mlkeys:
    excel = pd.read_excel('results_final/ml_subtype_pred/{0}.xlsx'.format(key),
                          index_col=0)
    mcc_mean, mcc_std = [], []
    columns.extend([key + '_mcc_mean', key + '_mcc_std'])
    for i in range(len(features)):
        mcc_mean.append(excel.mcc_mean[i])
        mcc_std.append(excel.mcc_std[i])
    store2.extend([mcc_mean, mcc_std])
store2 = pd.DataFrame(store2).transpose()
store2.index = features
store2.columns = columns

fnt = 'Arial'
ticks_font = {'fontname':fnt, 'size':'14'}
legend_font = {'family':fnt, 'size':'11'}
label_font = {'family':fnt, 'size':'18'}
plt.rcParams["figure.figsize"] = [6,3]
plt.rcParams['grid.alpha'] = 0.5

xlabel = features[:-1] + ['All-8']
color = ['dimgray', 'black', 'cadetblue', 'brown']
legend_label = ['Decision tree', 'Logistic regression', 'KNN', 'SVM']

X = np.arange(len(features))
lw = 1.0   # Width of bar edge
w = 0.175   #  Width of bar
cap = 2

out1 = plt.bar(X-1.5*w, store2.iloc[:,0], yerr=store2.iloc[:,1], 
               capsize=cap, color=color[0], width=w, linewidth=lw, 
               edgecolor='black')
out2 = plt.bar(X-0.5*w, store2.iloc[:,2], yerr=store2.iloc[:,3], 
               capsize=cap, color=color[1], width=w, linewidth=lw, 
               edgecolor='black')
out3 = plt.bar(X+0.5*w, store2.iloc[:,4], yerr=store2.iloc[:,5], 
               capsize=cap, color=color[2], width=w, linewidth=lw,
               edgecolor='black')
out4 = plt.bar(X+1.5*w, store2.iloc[:,6], yerr=store2.iloc[:,7], 
               capsize=cap, color=color[3], width=w, linewidth=lw,
               edgecolor='black')

plt.xlim(0-2*w,8+2*w)
plt.ylim([-1.0,1.20])
plt.yticks(**ticks_font)
plt.xticks(X, xlabel, rotation=0, **ticks_font)
plt.ylabel('MCC', **label_font)
pltout = [x[0] for x in [out1, out2, out3, out4]]
plt.legend(pltout, legend_label, frameon=1, numpoints=1, shadow=1, loc='best', 
           prop=legend_font)
plt.axhline(color='black')
plt.tight_layout()

plt.savefig('plots/mcc_plot.pdf')




# Outlier detection
#==========================#
looplength = pd.read_excel('results_final/looplength.xlsx', 
                           index_col=0)
subtype = pd.read_excel('results_final/cel7_subtypes.xlsx',
                        index_col=0)['ncbi_pred_class']
looplength.index = range(len(looplength))
subtype.index = range(len(subtype))

fnt='Arial'
ticks_font = {'fontname':fnt, 'size':'18'}
label_font = {'family':fnt, 'size':'22'}
legend_font = {'family':'Arial', 'size':'14'}
title_font = {'family':fnt, 'size':'30'}
plt.rcParams['figure.figsize'] = [6,4]

# View the distribution to intuitively determine outliers
maxlength = [14, 20, 25, 16, 52, 141, 50, 14]  # Values equal or greater than are outliers
topcode_vals = [] # Change the outlier values to top-coded values
for i in range(8):
    sortedvals = sorted(looplength.iloc[:,i])
    maxval = maxlength[i]
    topcode_vals.append(sortedvals[sortedvals.index(maxval) - 1])
    color = ['blue' if x<maxval else 'red' for x in sortedvals]
    loop = looplength.columns[i]
    plt.scatter(range(len(looplength)), sortedvals, color=color,
                marker='o')
    plt.xticks(**ticks_font)
    plt.yticks(**ticks_font)
    plt.xlabel('Index', **label_font)
    plt.ylabel('Length', **label_font)
    plt.title(loop, **title_font)
    plt.tight_layout()
    plt.savefig('plots/outlier_detection/{0}.pdf'.format(loop))
    plt.show()
    plt.close()

# Top-code outliers
looplength = looplength.iloc[:,:-1]
for i in range(len(looplength.columns)):
    vals = list(looplength.iloc[:,i])
    vals = [x if x<maxlength[i] else topcode_vals[i] for x in vals]
    looplength.iloc[:,i] = pd.Series(vals)




# Average loop lengths
#===============================#
cbh_looplength = looplength.iloc[subtype[subtype==1].index]
eg_looplength = looplength.iloc[subtype[subtype==0].index]
cbh_stat = cbh_looplength.describe()
eg_stat = eg_looplength.describe()
all_stat = looplength.describe()

fnt='Arial'
ticks_font = {'fontname':fnt, 'size':'16'}
label_font = {'family':fnt, 'size':'18'}
legend_font = {'family':'Arial', 'size':'14'}
title_font = {'family':fnt, 'size':'20'}
plt.rcParams['figure.figsize'] = [6,3]
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['axes.axisbelow'] = True
legend_label  = ['CBH', 'EG']

lw = 1.0   # Width of bar edge
w = 0.35   #  Width of bar
cap = 3
legend_label = ['CBH', 'EG']


X = np.arange(8)
color = ['lightblue', 'pink']
out1 = plt.bar(X-0.5*w, cbh_stat.loc['mean',:], color=color[0], width=w, 
               linewidth=lw, edgecolor='black', yerr=cbh_stat.loc['std',:],
               ecolor='black', capsize=cap)
out2 = plt.bar(X+0.5*w, eg_stat.loc['mean',:], color=color[1], width=w, 
               linewidth=lw, edgecolor='black', yerr=eg_stat.loc['std',:],
               ecolor='black', capsize=cap)
plt.ylim((0,18))
plt.xlim((-0.4,7.4))

plt.xticks(X, cbh_stat.columns, **ticks_font)
plt.yticks(np.arange(0,17,2),**ticks_font)
plt.ylabel('Number of residues', **label_font)
pltout = [x[0] for x in [out1, out2]]
plt.legend(pltout, legend_label, numpoints=1, loc='best', prop=legend_font)
plt.tight_layout()
plt.savefig('plots/looplengths.pdf')




# Relative standard deviation
#================================#
mean = np.mean(looplength, axis=0)
std = np.std(looplength, axis=0)
cov = std/mean*100

fnt='Arial'
ticks_font = {'fontname':fnt, 'size':'14'}
label_font = {'family':fnt, 'size':'15'}
plt.rcParams['figure.figsize'] = [6,3]

lw=1.3
plt.bar(range(len(cov)), cov, color='brown', linewidth=lw, 
        edgecolor='black')

plt.xticks(range(len(cov)), cov.index, **ticks_font)
plt.yticks([20,40,60,80,100], **ticks_font)
plt.xlim([-0.45,7.45])
plt.ylim([0,80])
plt.ylabel('Relative standard deviation (%)', **label_font)
plt.tight_layout()
plt.savefig('plots/rsd.pdf')




# Density plots (histogram) of loop lengths
#=============================================#
fnt='Arial'
ticks_font = {'fontname':fnt, 'size':'18'}
label_font = {'family':fnt, 'size':'22'}
legend_font = {'family':'Arial', 'size':'14'}
title_font = {'family':fnt, 'size':'30'}
plt.rcParams['figure.figsize'] = [6.5, 5]

bw = 0.5
xmin = [-0.5, -0.5, -0.5, -0.5, -0.5, -1.0, -0.5, -0.6]
xmax = [10, 16, 8, 13, 10, 11, 14, 8]
ymax = [0.5, 0.8, 0.8, 0.7, 0.8, 0.9, 0.5, 0.8]
legend_label = ['CBH', 'EG']
for i in range(len(looplength.columns)):
    col = looplength.columns[i]
    ax1 = sns.kdeplot(cbh_looplength[col], bw=bw, legend=True,
                      shade=False, color='blue')
    ax2 = sns.kdeplot(eg_looplength[col], bw=0.5, legend=True,
                      shade=False, color='red')
    ax1.legend(legend_label, loc='best', prop=legend_font)
    plt.xticks(**ticks_font)
    plt.yticks(np.arange(0,11,2)*0.1, **ticks_font)
    plt.xlim((0, xmax[i]))
    plt.ylim((0,ymax[i]))
    plt.title(col, **title_font)
    plt.xlabel('Number of residues', **label_font)
    plt.ylabel('Density', **label_font)
    plt.tight_layout()
    plt.savefig('plots/density_plots/{0}.pdf'.format(col))
    plt.show()
    plt.close()
    


    
# Correlation of loop lengths
#===============================================================================
p_corr, s_corr = [], []  # Pearson's and Spearman's correlation coefficients
for i in range(len(looplength.columns)):
    corr_p, corr_s = [], []
    for k in range(len(looplength.columns)):
        corr_p.append(np.corrcoef(looplength.iloc[:,i], 
                                  looplength.iloc[:,k])[0][1])
        corr_s.append(scipy.stats.spearmanr(looplength.iloc[:,i], 
                                            looplength.iloc[:,k])[0])
    p_corr.append(corr_p)
    s_corr.append(corr_s)
p_corr = pd.DataFrame(p_corr)
s_corr = pd.DataFrame(s_corr)
p_corr.index = looplength.columns
p_corr.columns = looplength.columns
s_corr.index = looplength.columns
s_corr.columns = looplength.columns

sns.set(font='Arial', font_scale=0.6)
cluster = sns.clustermap(p_corr, cmap='Reds', metric='euclidean', 
                         method='average', figsize=(3.15,3.15), 
                         annot=True, fmt='.2f', annot_kws={'size':6})
cluster.savefig('plots/looplength_corr.pdf')




# Classification/association rules table
#===========================================#
from get_rules import Cel7MSA
cbhmsa = 'fasta/trecel7a_positions_only/cbh_cat.fasta'
eglmsa = 'fasta/trecel7a_positions_only/egl_cat.fasta'
cel7msa = Cel7MSA(cbhmsa, eglmsa)
cel7msa.get_freq(include_gaps=True)
rules = pd.read_excel('results_final/rules/rules_all.xlsx', index_col=0)
rules_amino = pd.read_excel('results_final/rules/rules_amino.xlsx', index_col=0)
rules_type = pd.read_excel('results_final/rules/rules_type.xlsx', index_col=0)

mcc = list(rules.mcc)
min_mcc = np.percentile(mcc, 95)  # mcc > 0.73
rules_mcc = rules[rules.mcc >= min_mcc]
rules_amino_mcc = rules_amino[rules_amino.mcc >= min_mcc]  # 45 rules
rules_type_mcc = rules_type[rules_type.mcc >= min_mcc]  # 45 rules
positions = sorted(set(rules_mcc.tre_pos))  # 42 positions
rules_mcc.to_excel('results_final/rules/rules_mcc.xlsx')
rules_amino_mcc.to_excel('results_final/rules/rules_amino_mcc.xlsx')
rules_type_mcc.to_excel('results_final/rules/rules_type_mcc.xlsx')

rules_amino_table = rules_amino_mcc.loc[:,['tre_pos','rule', 'closest_subsite', 
                                           'dist_subsite','sens', 'spec', 'acc', 'mcc']]
rules_amino_table.columns = ['Position', 'Rule', 'Closest subsite', 
                             'Distance to closest subsite (Å)', 'Sensitivity', 
                             'Specificity', 'Accuracy', 'MCC']
rules_amino_table.to_excel('plots/rules_amino_table.xlsx')
rules_type_table = rules_type_mcc.loc[:,['tre_pos','rule', 'closest_subsite', 
                                         'dist_subsite', 'sens', 'spec', 'acc', 'mcc']]
rules_type_table.columns = ['Position', 'Rule', 'Closest subsite', 
                             'Distance to closest subsite (Å)', 'Sensitivity',
                             'Specificity', 'Accuracy', 'MCC']
rules_type_table.to_excel('plots/rules_type_table.xlsx')




# Plot Histogram for  MCC of rules
#=================================#
fnt='Arial'
ticks_font = {'fontname':fnt, 'size':'20'}
label_font = {'family':fnt, 'size':'22'}
title_font = {'family':fnt, 'size':'24'}
plt.rcParams['figure.figsize'] = [6,3.5]
plt.rcParams['grid.alpha'] = 0.5
plt.rcParams['axes.axisbelow'] = True

weights = np.zeros_like(mcc) + 1/len(mcc)
plt.hist(mcc, bins=12, rwidth=1, color='darkgreen', weights=weights)
plt.xticks(np.arange(-80,101,40)*0.01, **ticks_font)
plt.yticks(np.arange(0,28,5)*0.01, **ticks_font)
plt.xlabel('MCC', **label_font)
plt.ylabel('Relative frequency', **label_font)
plt.tight_layout()
plt.savefig('plots/rules_mcc_dist.pdf')




# Minimum distance between rules' positions and substrate
#============================================================#
dist50 = np.percentile(rules_mcc.dist_subsite, 50) #4.79Å
rule_dist = list(rules_mcc.dist_subsite)
weights = np.zeros_like(rule_dist) + 1/len(rule_dist)
plt.hist(rule_dist, bins=7, weights=weights, color='indigo')
plt.xticks(np.arange(0,30,5), **ticks_font)
plt.xlim((0,25))
plt.yticks(**ticks_font)
plt.xlabel('Distance to substrate (Å)', **label_font)
plt.ylabel('Relative frequency', **label_font)
plt.tight_layout()
plt.savefig('plots/rules_distance_dist.pdf')




# Distribution at position 40
#===========================#
cel7msa.site_plot(site=40, savefig=True)




# Distribution at 42 important positions
#==========================================#
plt.rcParams['figure.figsize'] = [7,4]
for i in range(len(positions)):
    cel7msa.site_plot(site=positions[i], savefig=True)




# Pymol selection command to visualize rules on structure
#=========================================================#
pymol_positions = 'select rules, ('
for pos in positions:
    pymol_positions += 'resi {0} or '.format(pos)
pymol_positions = pymol_positions[:-4]
pymol_positions += ') and name ca'
with open('plots/rules_pymol.txt', 'w') as txt:
    txt.write(pymol_positions)




# Aromatic residues within 6Å of substrate (and consensus CBH and EGL)
#==============================================================================#
cel7msa.get_consensus_sequences()
cbh_consensus = list(cel7msa.consensus_cbh)
egl_consensus = list(cel7msa.consensus_egl)
tre = bioinf.split_fasta('fasta/trecel7a_positions_only/consensus.fasta')[1][1]
excel = pd.read_excel('results_final/residue_distances.xlsx', index_col=0)
closest_subsite = list(excel.iloc[:,0])
distances = list(excel.iloc[:,1])

resid_aro, tre_aro, cbh_aro, egl_aro, closest_subsite_aro, dist_aro = [],[],[],[],[],[]
cbh_aro_freq, egl_aro_freq, conserved = [], [], []
aro_res = ['F', 'W', 'Y', 'H']

for i in range(len(tre)):
    if (tre[i] in aro_res or cbh_consensus[i] in  aro_res or egl_consensus[i] in aro_res)\
    and distances[i]<=6.0:
        resid_aro.append(i+1)
        tre_aro.append(tre[i])
        cbh_aro.append(cbh_consensus[i])
        egl_aro.append(egl_consensus[i])
        closest_subsite_aro.append(closest_subsite[i])
        dist_aro.append(distances[i])
        cbh_freq = cel7msa.cbh_freq.iloc[[4,6,18,19],i].sum()*100
        egl_freq = cel7msa.egl_freq.iloc[[4,6,18,19],i].sum()*100
        cbh_aro_freq.append(cbh_freq)
        egl_aro_freq.append(egl_freq)
        if cbh_freq > 66 and egl_freq < 66:
            conserved.append('CBH')
        elif cbh_freq < 66 and egl_freq > 66:
            conserved.append('EG')
        elif cbh_freq > 66 and egl_freq > 66:
            conserved.append('CBH and EG')
        else:
            conserved.append('None')

store = pd.DataFrame([resid_aro, tre_aro, cbh_aro, egl_aro, cbh_aro_freq, egl_aro_freq, 
                      closest_subsite_aro, dist_aro, conserved]).transpose()
store.columns = ['Position', 'Trecel7A residue', 'CBH consensus residue', 
                 'EG consensus residue', 'Frequency of aromatic residues in CBHs (%)', 
                 'Frequency of aromatic residues in EGs (%)', 'Closest subsite', 
                 'Distance to closest subsite (Å)', 'Aromatic residues conserved (>66%) in']
store = store.sort_values('Closest subsite')
store.to_excel('results_final/aromatic_residues.xlsx')




# Pymol commands for viewing aromatic residues on structure
#=============================================================#
pymol_cbh = 'select arocbh, '
pymol_both = 'select aroboth, '
for i in range(len(store)):
    pos = store.iloc[i,0]
    if store.iloc[i,-1]=='CBH':
        pymol_cbh += 'resi {0} or '.format(pos)
    elif store.iloc[i,-1]=='CBH and EG':
        pymol_both += 'resi {0} or '.format(pos)
with open('plots/aromatic_pymol.txt', 'w') as pym:
    pym.write(pymol_cbh[:-4] + '\n\n')
    pym.write(pymol_both[:-4] + '\n\n')

pymol_positions = 'select rules, ('
for pos in positions:
    pymol_positions += 'resi {0} or '.format(pos)
pymol_positions = pymol_positions[:-4]
pymol_positions += ') and name ca'
with open('plots/rules_pymol.txt', 'w') as txt:
    txt.write(pymol_positions)




# Feature importances for CBM prediction (All 5933 features)
#==============================================================#
ex = pd.read_excel('results_final/ml_cbm_pred/featimp_all.xlsx', index_col=0)
ex = ex.sort_values('mean', ascending=False)

fnt='Arial'
ticks_font = {'fontname':fnt, 'size':'16'}
label_font = {'family':fnt, 'size':'20'}
plt.rcParams['figure.figsize'] = [6,3]

plt.bar(range(len(ex)), list(ex.iloc[:,1]), color='maroon', linewidth=1, edgecolor='maroon')
plt.xticks(**ticks_font)
plt.yticks(**ticks_font)
plt.xlabel('Feature No.', **label_font)
plt.ylabel('Gini importance', **label_font)
plt.tight_layout()
plt.savefig('plots/cbm_all_featimp.pdf')




# Plot of feature importances of top 20 features
#================================================#
ex = pd.read_excel('results_final/ml_cbm_pred/featimp_top20.xlsx', index_col=0)
ex = ex.sort_values('mean', ascending=False)

fnt='Arial'
ticks_font = {'fontname':fnt, 'size':'16'}
label_font = {'family':fnt, 'size':'20'}
plt.rcParams['figure.figsize'] = [6,3]

plt.bar(range(len(ex)), ex.iloc[:,1], color='maroon', linewidth=0.6, edgecolor='black',
        yerr=ex.iloc[:,2], ecolor='black', capsize=3)
plt.xticks(range(len(ex)), ex.iloc[:,0], **ticks_font, rotation=90)
plt.yticks(**ticks_font)
plt.xlabel('Features', **label_font)
plt.ylabel('Gini importance', **label_font)
plt.tight_layout()
plt.savefig('plots/cbm_top20_featimp.pdf')




# Pymol commands for viewing top 20 positions on structure
#===========================================================#
positions = list(ex.iloc[:,0])
positions = [x[1:] for x in positions]
pymol_positions = 'select top20, ('
for pos in positions:
    pymol_positions += 'resi {0} or '.format(pos)
pymol_positions = pymol_positions[:-4]
pymol_positions += ') and protein'
with open('plots/cbm_pymol.txt', 'w') as txt:
    txt.write(pymol_positions)




# Table of position-specific rules for predicting CBM
#================================================#
ex = pd.read_excel('results_final/ml_cbm_pred/position_rules.xlsx', index_col=0)
table = pd.DataFrame()
table['position'] = [int(x.split()[-1].split('=>')[0][1:]) for x in ex.rule]
table['rule'] = ex.rule
table['sensitivity'] = [round(x, 1) for x in ex.sensitivity]
table['specificity'] = [round(x,1) for x in ex.specificity]
table['accuracy'] = [round(x,1) for x in ex.accuracy]
table['mcc'] = [round(x,2) for x in ex.mcc]
table = table.sort_values('position', ascending=True)
table.to_excel('plots/cbm_rules.xlsx')



