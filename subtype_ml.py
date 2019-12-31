#===============================================================#
# Prediction of GH7 subtypes with machine learning (ML)
#===============================================================#




# Imports
#===============================#

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import pydot_ng as pydot

from imblearn.under_sampling import RandomUnderSampler

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.externals.six import StringIO

import warnings
warnings.filterwarnings("ignore")

import bioinformatics as bioinf




# Get lengths of loops from MSA
#================================================#
def get_gh7looplength(msafasta, trecel7a_pos=0):
    ''' Return a DataFrame of the number of residues
    in the 8 loops of GH7 sequences in an MSA fasta file.
    TreCel7A is used as reference for determining the loop 
    positions in the MSA. The position of TreCel7A in the 
    fasta file is trecel7a_pos (0 if first). Loop lengths 
    are in the order [A1, A2, A3, A4, B1, B2, B3, B4]. '''
    
    # Loop residues in TreCel7A
    loopres = ['QSAQK', 'TSSGVPAQVESQS', 'DYYAN', 'TNETSSTPGA',
               'YDGNTW', 'PSSNNANT', 'GGTYSDNRYG', 'GGSS']  # Residues in the loops of TreCel7A
    loopmore = ['NVGARLY', 'PNAKVTFSNIK', 'MLWLDST', 'VRGSCSTSSGVPA',
                'SSTLCPD', 'GIGGHGSCCS', 'GTCDPDGCDWNP', 'FSDKGGL']  # Residues after the loops
    
    # Get aligned sequences
    [heads, sequences] = bioinf.split_fasta(msafasta)   # Retrieve sequences from fasta file 
    trecel7a_seq_msa = sequences[trecel7a_pos]
    trecel7a_nogaps = trecel7a_seq_msa.replace('-','')
    trecel7a_list = list(trecel7a_seq_msa)
    
    # Get loop positions in MSA (using TreCel7A as reference)
    numb = -1
    for k in range(len(trecel7a_list)):
        if trecel7a_list[k].isalpha():
            numb += 1
            trecel7a_list[k] = str(numb)
    startpos = [trecel7a_list.index(str(trecel7a_nogaps.index(loopres[i]))) 
                    for i in range(len(loopres))]
    stoppos = [trecel7a_list.index(str(trecel7a_nogaps.index(loopmore[i]))) 
                    for i in range(len(loopmore))]
    length = [stoppos[i] - startpos[i] for i in range(len(startpos))]
    
    # Determine  loop length
    store = []
    for i in range(len(sequences)):
        seq = sequences[i]
        loopregion = [seq[startpos[k]:stoppos[k]] for k in range(len(loopres))]
        looplength = [length[k] - loopregion[k].count('-') for k in range(len(loopres))]
        store.append(looplength)
        
    # Save results as DataFrame
    result = pd.DataFrame(store)
    result.columns = ['A1', 'A2', 'A3', 'A4', 'B1', 'B2', 'B3', 'B4']
    return result


# Calculate loop lengths
msafile = 'fasta/structure_based_alignment/cel7_nr99_structaln.fasta'
looplength = get_gh7looplength(msafile, trecel7a_pos=0)


# Write results to excel spreadhseet
looplength.index = range(1, len(looplength)+1)
looplength['accession'] = bioinf.get_accession(msafile)
looplength.to_excel('results_final/looplength.xlsx')




# Data preprocessing: prepare data for machine learning
#================================================================#

# Retreive data
looplength = pd.read_excel('results_final/looplength.xlsx', index_col=0)
subtype = pd.read_excel('results_final/cel7_subtypes.xlsx', index_col=0)
looplength.index = range(len(looplength))
subtype.index = range(len(subtype))
assert looplength.accession.equals(subtype.accession)  # Ensure sequence positions are the same

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
                marker='.')
    plt.xlabel('Index')
    plt.ylabel('Length')
    plt.title(loop)
    plt.show()
    plt.close()


# Deal with outliers
# Top-coding outliers before calculating Z-scores
X_grand = looplength.iloc[:,:-1]
for i in range(len(X_grand.columns)):
    vals = list(X_grand.iloc[:,i])
    vals = [x if x<maxlength[i] else topcode_vals[i] for x in vals]
    X_grand.iloc[:,i] = pd.Series(vals)
    

# Standardize data (convert to Z-scores)
scx = StandardScaler()
X_grand = pd.DataFrame(scx.fit_transform(X_grand))
X_grand.columns = looplength.iloc[:,1:].columns
y_grand = pd.Series(list(subtype['ncbi_pred_class']), index=range(len(subtype)))


    


# Apply machine learning to predict subtypes from loop lengths
#==============================================================================#
def get_classifier(clf_type, depth=1):
    '''Return an instance of the specified classifier.'''
    if clf_type == 'dec':
        classifier = DecisionTreeClassifier(criterion='entropy', max_depth=depth)
    elif clf_type == 'svm':
        classifier = SVC(kernel='rbf')
    elif clf_type == 'knn':
        classifier = KNeighborsClassifier(n_neighbors=10)
    elif clf_type == 'log':
        classifier = LogisticRegression()
    return classifier


def apply_ML(X_grand, y_grand, clf_type, monte_count=100):
    '''Apply ML to predict subtypes from loop lengths.
    Return a tuple of dataframes of performance results,
    (sensitivity, specificity, accuracy, MCC).'''
    
    # Empty lists for storing final results
    sens_final, spec_final, acc_final, mcc_final = [], [], [], []
    
    # Monte Carlo loop
    for i in range(monte_count):
        RUS = RandomUnderSampler(random_state=None) # Random undersampling of majority class
        X_select, y_select = RUS.fit_resample(X_grand, y_grand)
        X_select, y_select = pd.DataFrame(X_select), pd.Series(y_select)
            
        # K-fold cross validation
        kf = KFold(n_splits=5, shuffle=True, random_state=None)
        kf_indices = kf.split(X_select)
        for train_index, test_index in kf_indices:
            X_train, y_train = X_select.iloc[train_index,:], y_select.iloc[train_index]
            X_test, y_test = X_select.iloc[test_index,:], y_select.iloc[test_index]
            
            # Empty lists for storing kfold cross validation results
            sens_store, spec_store, acc_store, mcc_store = [], [], [], []
            
            # Single-feature loop (Train classifier using single feature independently)
            for j in range(8):
                # Get classifier and fit to training data
                classifier = get_classifier(clf_type)
                classifier.fit(pd.DataFrame(X_train.iloc[:,j]), y_train)
                     
                # Test classifier on test set
                y_pred = classifier.predict(pd.DataFrame(X_test.iloc[:,j]))       
                
                # Evaluate performance
                cm = confusion_matrix(y_test, y_pred)
                tn, tp, fn, fp = cm[0][0], cm[1][1], cm[1][0], cm[0][1]
                n = tp + fp + tn + fn
                accuracy = (tp + tn)/n * 100
                mcc = ((tp*tn) - (fp*fn))/np.sqrt((tp+fp)*(tn+fn)*(tp+fp)*(tn+fp))
                sens = tp/(tp + fn) * 100 if tp + fp != 0 else 0
                spec = tn/(tn + fp) * 100 if tn + fn != 0 else 0
                
                # Save results
                sens_store.append(sens)
                spec_store.append(spec)
                acc_store.append(accuracy)
                mcc_store.append(mcc)
                
            # Multiple-features (Train classifier on all features)
            classifier = get_classifier(clf_type, depth=8)
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(pd.DataFrame(X_test))
            
            # Evaluate performance
            cm = confusion_matrix(y_test, y_pred)
            tn, tp, fn, fp = cm[0][0], cm[1][1], cm[1][0], cm[0][1]
            n = tp + fp + tn + fn
            accuracy = (tp + tn)/n * 100
            mcc = ((tp*tn) - (fp*fn))/np.sqrt((tp+fp)*(tn+fn)*(tp+fp)*(tn+fp))
            sens = tp/(tp + fn) * 100 if tp + fp != 0 else 0
            spec = tn/(tn + fp) * 100 if tn + fn != 0 else 0
            
            # Save results
            sens_store.append(sens)
            spec_store.append(spec)
            acc_store.append(accuracy)
            mcc_store.append(mcc)
            
            # Save all results to final store
            sens_final.append(sens_store)
            spec_final.append(spec_store)
            acc_final.append(acc_store)
            mcc_final.append(mcc_store)
    
    sens_final = pd.DataFrame(sens_final)
    spec_final = pd.DataFrame(spec_final)
    acc_final = pd.DataFrame(acc_final)
    mcc_final = pd.DataFrame(mcc_final)
    
    # Combine results to a single dataframe
    results = pd.DataFrame()
    columns = ['A1', 'A2', 'A3', 'A4', 'B1','B2', 'B3', 'B4', 'all8']
    results['features'] = columns
    results['sens_mean'] = sens_final.mean()
    results['sens_std'] = sens_final.std()
    results['spec_mean'] = spec_final.mean()
    results['spec_std'] = spec_final.std()
    results['acc_mean'] = acc_final.mean()
    results['acc_std'] = acc_final.std()
    results['mcc_mean'] = mcc_final.mean()
    results['mcc_std'] = mcc_final.std()
    
    return results
        

# Implement machine learning using 4 different classifiers
clf_types = ['dec', 'svm', 'knn', 'log']
for clf_type in clf_types:
    results = apply_ML(X_grand, y_grand, clf_type, monte_count=100)
    results.to_excel('results_final/ml_subtype_pred/{0}.xlsx'.format(clf_type))


                

# Get decision tree rules
#=============================#
X_grand = looplength.iloc[:,1:] # Non-standardized lengths
for i in range(len(X_grand.columns)):
    RUS = RandomUnderSampler(random_state=None)
    X = pd.DataFrame(X_grand.iloc[:,i])
    loop = X.columns
    X, y = RUS.fit_resample(X, y_grand)
    X, y = pd.DataFrame(X,columns=loop ), pd.Series(y)
    clf = DecisionTreeClassifier(max_depth=1, criterion='entropy')
    clf.fit(X, y)
    dot_data = StringIO()
    tree.export_graphviz(clf, out_file=dot_data,
                         feature_names=X.columns,
                         class_names=['EG', 'CBH'],
                         filled=True, rounded=True,
                         special_characters=True)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf('plots/dec_tree_rules/{0}.pdf'.format(X.columns[0]))
    



# Probability of significant truncation in  B2, B3, and B4 loops
# given that A4 loop is truncated
#==============================================================================#

X_grand = looplength.iloc[:,1:] # Non-standardized lengths
#a4_less6 = X_grand[X_grand['A4']<=5]
b4_less = X_grand[X_grand['B4']<=3]
all_less = b4_less[b4_less['B3']<=3]
all_less = all_less[all_less['A4']<=5]
all_less = all_less[all_less['B2']<=4]
proba = len(all_less)/len(b4_less) * 100


