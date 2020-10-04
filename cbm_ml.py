"""
Predict presence of CBM from catalytic domain residues with machine learning
"""




# Imports
#=====================#
import pandas as pd
import numpy as np
from scipy import stats
import random

import os
import subprocess

from imblearn.under_sampling import RandomUnderSampler

from Bio.Blast.Applications import NcbiblastpCommandline

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

import warnings
warnings.filterwarnings("ignore")

import bioinformatics as bioinf






# Determine presence of CBM from full length sequences
#=====================================================#

# Make database of 1,748 sequences 
# (NCBI blast executable should be installed on your computer)
directory = 'cbm_database'
if not os.path.exists(directory):
    os.mkdir(directory)
makeblast_exe = '/usr/local/ncbi/blast/bin/makeblastdb' # Path of NCBI makeblastdb executable (CHANGE THIS!)
fasta_file = 'fasta/initial_blast/cel7_nr99_full_length.fasta'
subprocess.call(f'{makeblast_exe} -in {fasta_file} -dbtype prot -out cel7_cbm', shell=True)
subprocess.call('mv cel7_cbm.* cbm_database/', shell=True)


# Blast TreCel7A CBM against 1748 sequences
# (NCBI blast executable should be installed on your computer)
blastp_exe = '/usr/local/ncbi/blast/bin/blastp'   # Path of blastp executable (CHANGE THIS)
trecel7a_cbm = 'cbm_database/trecel7a_cbm1.fasta'
output = 'results_final/cbm_blast_output.txt'
blast_cline = NcbiblastpCommandline(cmd=blastp_exe, query=trecel7a_cbm, 
                                    db='cbm_database/cel7_cbm', evalue=1e-3,
                                    outfmt=7, num_alignments=2000, out=output)
stdout, stderr = blast_cline()  # evalue of 1e-3 or less corresponds to bit score of 30 or more


# CBM data for all sequences
ex = pd.read_csv('results_final/cbm_blast_output.csv')  # csv derived from txt file output
accession_cbm = list(ex['subject'])
accession_all = bioinf.get_accession('fasta/initial_blast/cel7_nr99_full_length.fasta')
has_cbm = [1 if x in accession_cbm else 0 for x in accession_all]
df = pd.DataFrame([accession_all, has_cbm], index=['accession', 'has_cbm']).transpose()
df.to_csv('results_final/has_cbm.csv')


# CBM distribution
df['subtype'] = pd.read_excel('results_final/cel7_subtypes.xlsx')['ncbi_pred_class']
df_cbh = df[df['subtype']==1]
df_egl = df[df['subtype']==0]
cbh_cbm = df_cbh.has_cbm.value_counts()[1]
cbh_nocbm = df_cbh.has_cbm.value_counts()[0]
egl_cbm = df_egl.has_cbm.value_counts()[1]
egl_nocbm = df_egl.has_cbm.value_counts()[0]






# Derive features for machine learning with one-hot encoding
#============================================================#
cat_domain_fasta = 'fasta/trecel7a_positions_only/cel7_cat.fasta'
sequence_df = bioinf.fasta_to_df(cat_domain_fasta)
X_features = pd.DataFrame() # empty dataframe for storing features

for i in range(len(sequence_df.columns)):
    # Convert amino acids to integers
    X_resid = list(sequence_df.iloc[:,i])
    labelencoder = LabelEncoder()
    X_label = list(labelencoder.fit_transform(X_resid))
    X_resid_unique = sorted(set(X_resid))
    X_label_unique = sorted(set(X_label))
    
    # Map integer labels to amino acids
    label_resid = [X_label.index(num) for num in X_label_unique]
    label_resid = [X_resid[num] for num in label_resid]
    
    # Convert labels to binary features (one-hot encoding)
    onehotencoder = OneHotEncoder()
    X_label = pd.DataFrame(X_label) # convert to 2D array
    X_encoded = onehotencoder.fit_transform(X_label).toarray()
    X_encoded = pd.DataFrame(X_encoded)
    
    # Name encoded features (residue + position, e.g G434)
    X_encoded.columns = ['{0}{1}'.format(res,i+1) for res in label_resid]
    del X_encoded['-{0}'.format(i+1)]  # remove encoded features from gaps
    
    # Append features to dataframe store
    for col in X_encoded.columns:
        X_features[col] = X_encoded[col]    
    
    




# Randomly split data to validation set and test set
#====================================================#
y = pd.Series(has_cbm)   # class labels
y_yescbm = y[y==1]  
y_nocbm = y[y==0]

# Test set data (10% of total data)
yes_test_size = int(0.1 * len(y_yescbm))
no_test_size = int(0.1 * len(y_nocbm))
yes_test_indices = random.sample(list(y_yescbm.index), yes_test_size)
no_test_indices = random.sample(list(y_nocbm.index), no_test_size)
test_indices = yes_test_indices + no_test_indices
test_indices = sorted(test_indices)

# Validation set data (90% of total data)
val_indices = [x for x in list(y.index) if x not in test_indices]

# X (features) and y for validation and test sets
X_val = X_features.iloc[val_indices,:]
y_val = y.iloc[val_indices]
X_test_sep = X_features.iloc[test_indices,:]
y_test_sep = y.iloc[test_indices]






# Apply random forests to validation set using all features
#=============================================================#

# Empty lists for storing final results
sens_store, spec_store, acc_store, mcc_store, featimp_store = [], [], [], [], []

# Function for evaluating performance
def evalPerf(y_test, y_pred):
    '''Return (sensitivity, specificity, accuracy, MCC, p_value)'''
    cm = confusion_matrix(y_test, y_pred)
    tn, tp, fn, fp = cm[0][0], cm[1][1], cm[1][0], cm[0][1]
    n = tp + fp + tn + fn
    accuracy = (tp + tn)/n * 100
    mcc = ((tp*tn) - (fp*fn))/np.sqrt((tp+fp)*(tn+fn)*(tp+fp)*(tn+fp))
    sens = tp/(tp + fn) * 100 if tp + fp != 0 else 0
    spec = tn/(tn + fp) * 100 if tn + fn != 0 else 0
    table = np.array([[tp, fp], [fn, tn]]) # CBH and EG have same contingency table
    p_value = stats.chi2_contingency(table)[1]
    return [sens, spec, accuracy, mcc, p_value]


# 100 repetitions of 5-fold cross validation
for r in range(100):
    RUS = RandomUnderSampler(random_state=None)
    X_select, y_select = RUS.fit_resample(X_val, y_val)
    X_select, y_select = pd.DataFrame(X_select), pd.Series(y_select)
    
    # 5-fold cross validation
    kf = KFold(n_splits=5, shuffle=True, random_state=None)
    kf_indices = kf.split(X_select)
    for train_index, test_index in kf_indices:
        X_train, y_train = X_select.iloc[train_index, :], y_select.iloc[train_index]
        X_test, y_test = X_select.iloc[test_index, :], y_select.iloc[test_index]
        
        # Fit random forest classifier to training data
        classifier = RandomForestClassifier(n_estimators=500, n_jobs=-1)
        classifier.fit(X_train, y_train)
        
        # Test classifier and evaluate performance
        y_pred = classifier.predict(X_test)
        sens, spec, accuracy, mcc, pvalue = evalPerf(y_test, y_pred)
        featimp = list(classifier.feature_importances_)
        
        # Save results
        sens_store.append(sens)
        spec_store.append(spec)
        acc_store.append(accuracy)
        mcc_store.append(mcc)
        featimp_store.append(featimp)

# Average results over all 500 repetitions
store = [np.mean(sens_store), np.std(sens_store), np.mean(spec_store), np.std(spec_store),
         np.mean(acc_store), np.std(acc_store), np.mean(mcc_store), np.std(mcc_store)]
store = pd.DataFrame(store, index=['sens_mean', 'sens_std', 'spec_mean', 'spec_std',
                                   'acc_mean', 'acc_std', 'mcc_mean', 'mcc_std'])
featimp_mean = pd.DataFrame(featimp_store).mean(axis=0)
featimp_std = pd.DataFrame(featimp_store).std(axis=0)
store_featimp = pd.DataFrame([X_val.columns, featimp_mean, featimp_std], 
                             index=['features', 'mean', 'std']).transpose()

# Write results to spreadsheet
store.to_csv('results_final/ml_cbm_pred/perf_all.csv')
store_featimp.to_csv('results_final/ml_cbm_pred/featimp_all.csv')
     
   
        


        
# Use only top 50 features
#===================================#

# Top 50 features
top50_index = list(store_featimp.sort_values(by='mean', ascending=False).iloc[:50,:].index)
X_val_top50 = X_val.iloc[:,top50_index]

# Empty lists for storing final results
sens_store, spec_store, acc_store, mcc_store, featimp_store = [], [], [], [], []

# 100 repetitions of 5-fold cross validation
for r in range(100):
    RUS = RandomUnderSampler(random_state=None)
    X_select, y_select = RUS.fit_resample(X_val_top50, y_val)
    X_select, y_select = pd.DataFrame(X_select), pd.Series(y_select)
    
    # 5-fold cross validation
    kf = KFold(n_splits=5, shuffle=True, random_state=None)
    kf_indices = kf.split(X_select)
    for train_index, test_index in kf_indices:
        X_train, y_train = X_select.iloc[train_index, :], y_select.iloc[train_index]
        X_test, y_test = X_select.iloc[test_index, :], y_select.iloc[test_index]
        
        # Fit random forest classifier to training data
        classifier = RandomForestClassifier(n_estimators=500, n_jobs=-1)
        classifier.fit(X_train, y_train)
        
        # Test classifier and evaluate performance
        y_pred = classifier.predict(X_test)
        sens, spec, accuracy, mcc, pvalue = evalPerf(y_test, y_pred)
        featimp = list(classifier.feature_importances_)
        
        # Save results
        sens_store.append(sens)
        spec_store.append(spec)
        acc_store.append(accuracy)
        mcc_store.append(mcc)
        featimp_store.append(featimp)

# Average results over all 500 repetitions
store_top50 = [np.mean(sens_store), np.std(sens_store), np.mean(spec_store), np.std(spec_store),
         np.mean(acc_store), np.std(acc_store), np.mean(mcc_store), np.std(mcc_store)]
store_top50 = pd.DataFrame(store_top50, index=['sens_mean', 'sens_std', 'spec_mean', 'spec_std',
                                   'acc_mean', 'acc_std', 'mcc_mean', 'mcc_std'])
featimp_mean_top50 = pd.DataFrame(featimp_store).mean(axis=0)
featimp_std_top50 = pd.DataFrame(featimp_store).std(axis=0)
store_featimp_top50 = pd.DataFrame([X_val_top50.columns, featimp_mean_top50, featimp_std_top50], 
                             index=['features', 'mean', 'std']).transpose()

# Write results to spreadsheet
store_top50.to_csv('results_final/ml_cbm_pred/perf_top50.csv')
store_featimp_top50.to_csv('results_final/ml_cbm_pred/featimp_top50.csv')






# Repeat random forest without on top 50 features without terminal residues 
#=============================================================================#

# Top 50 features
top50_index = list(store_featimp.sort_values(by='mean', ascending=False).iloc[:50,:].index)
X_val_top50_all = X_val.iloc[:,top50_index]

# Remove features from positions 431, 432, 433 and 434
X_val_top50 = pd.DataFrame()
for col in X_val_top50_all.columns:
    if int(col[1:])<424:
        X_val_top50[col] = X_val_top50_all[col]

# Empty lists for storing final results
sens_store, spec_store, acc_store, mcc_store, featimp_store = [], [], [], [], []

# 100 repetitions of 5-fold cross validation
for r in range(100):
    RUS = RandomUnderSampler(random_state=None)
    X_select, y_select = RUS.fit_resample(X_val_top50, y_val)
    X_select, y_select = pd.DataFrame(X_select), pd.Series(y_select)
    
    # 5-fold cross validation
    kf = KFold(n_splits=5, shuffle=True, random_state=None)
    kf_indices = kf.split(X_select)
    for train_index, test_index in kf_indices:
        X_train, y_train = X_select.iloc[train_index, :], y_select.iloc[train_index]
        X_test, y_test = X_select.iloc[test_index, :], y_select.iloc[test_index]
        
        # Fit random forest classifier to training data
        classifier = RandomForestClassifier(n_estimators=500, n_jobs=-1)
        classifier.fit(X_train, y_train)
        
        # Test classifier and evaluate performance
        y_pred = classifier.predict(X_test)
        sens, spec, accuracy, mcc, pvalue = evalPerf(y_test, y_pred)
        featimp = list(classifier.feature_importances_)
        
        # Save results
        sens_store.append(sens)
        spec_store.append(spec)
        acc_store.append(accuracy)
        mcc_store.append(mcc)
        featimp_store.append(featimp)

# Average results over all 500 repetitions
store_top50_noterm = [np.mean(sens_store), np.std(sens_store), np.mean(spec_store), np.std(spec_store),
         np.mean(acc_store), np.std(acc_store), np.mean(mcc_store), np.std(mcc_store)]
store_top50_noterm = pd.DataFrame(store_top50_noterm, index=['sens_mean', 'sens_std', 'spec_mean', 'spec_std',
                                   'acc_mean', 'acc_std', 'mcc_mean', 'mcc_std'])
featimp_mean_top50 = pd.DataFrame(featimp_store).mean(axis=0)
featimp_std_top50 = pd.DataFrame(featimp_store).std(axis=0)
store_featimp_top50 = pd.DataFrame([X_val_top50.columns, featimp_mean_top50, featimp_std_top50], 
                             index=['features', 'mean', 'std']).transpose()

# Write results to  spreadsheet
store_top50.to_csv('results_final/ml_cbm_pred/perf_top50_noterm.csv')






# Use only top 20 features
#=================================#

# Top 20 features
top20_index = list(store_featimp.sort_values(by='mean', ascending=False).iloc[:20,:].index)
X_val_top20 = X_val.iloc[:,top20_index]

# Empty lists for storing final results
sens_store, spec_store, acc_store, mcc_store, featimp_store = [], [], [], [], []

# 100 repetitions of 5-fold cross validation
for r in range(100):
    RUS = RandomUnderSampler(random_state=None)
    X_select, y_select = RUS.fit_resample(X_val_top20, y_val)
    X_select, y_select = pd.DataFrame(X_select), pd.Series(y_select)
    
    # 5-fold cross validation
    kf = KFold(n_splits=5, shuffle=True, random_state=None)
    kf_indices = kf.split(X_select)
    for train_index, test_index in kf_indices:
        X_train, y_train = X_select.iloc[train_index, :], y_select.iloc[train_index]
        X_test, y_test = X_select.iloc[test_index, :], y_select.iloc[test_index]
        
        # Fit random forest classifier to training data
        classifier = RandomForestClassifier(n_estimators=500, n_jobs=-1)
        classifier.fit(X_train, y_train)
        
        # Test classifier and evaluate performance
        y_pred = classifier.predict(X_test)
        sens, spec, accuracy, mcc, pvalue = evalPerf(y_test, y_pred)
        featimp = list(classifier.feature_importances_)
        
        # Save results
        sens_store.append(sens)
        spec_store.append(spec)
        acc_store.append(accuracy)
        mcc_store.append(mcc)
        featimp_store.append(featimp)

# Average results over all 500 repetitions
store_top20 = [np.mean(sens_store), np.std(sens_store), np.mean(spec_store), np.std(spec_store),
         np.mean(acc_store), np.std(acc_store), np.mean(mcc_store), np.std(mcc_store)]
store_top20 = pd.DataFrame(store_top20, index=['sens_mean', 'sens_std', 'spec_mean', 'spec_std',
                                   'acc_mean', 'acc_std', 'mcc_mean', 'mcc_std'])
featimp_mean_top20 = pd.DataFrame(featimp_store).mean(axis=0)
featimp_std_top20 = pd.DataFrame(featimp_store).std(axis=0)
store_featimp_top20 = pd.DataFrame([X_val_top20.columns, featimp_mean_top20, featimp_std_top20], 
                             index=['features', 'mean', 'std']).transpose()

# Write results to excel spreadsheet
store_top20.to_csv('results_final/ml_cbm_pred/perf_top20.csv')
store_featimp_top20.to_csv('results_final/ml_cbm_pred/featimp_top20.csv')
        



# Train top20 classifier and test on test set
#===============================================#
RUS = RandomUnderSampler(random_state=None)
X_select, y_select = RUS.fit_resample(X_val_top20, y_val)
classifier = RandomForestClassifier(n_estimators=500, n_jobs=-1)
classifier.fit(X_select, y_select)
y_pred = classifier.predict(X_test_sep.iloc[:,top20_index])
cm = confusion_matrix(y_test_sep, y_pred)
tn, tp, fn, fp = cm[0][0], cm[1][1], cm[1][0], cm[0][1]
sens, spec, accuracy, mcc, pvalue = evalPerf(y_test_sep, y_pred)
store = pd.DataFrame([tp, fp, tn, fn, sens, spec, accuracy, mcc], 
                     index=['tp', 'fp', 'tn', 'fn', 'sens', 'spec', 'acc', 'mcc'])
store.to_csv('results_final/ml_cbm_pred/perf_test_set.csv')

        




# Position specific rules from top 50 features (Supporting Information)
#===============================================================================#
store = [] # empty list for storing results
for col in X_val_top50.columns:
    # Test the rule, feature => CBM
    y_pred = X_val_top50[col]
    cm = confusion_matrix(y_val, y_pred)
    perf = evalPerf(y_val, y_pred)
    
    # Test the rule, not feature => CBM
    y_pred1 = [1 if x==0 else 0 for x in y_pred]
    cm1 = confusion_matrix(y_val, y_pred1)
    perf1 = evalPerf(y_val, y_pred1)
    
    # Add rule (i.e. the rule with the highest MCC)
    if perf[3] > perf1[3]:
        perf = ['{0}=>CBM'.format(col)] + perf
        store.append(perf)
    else:
        perf1 = ['not {0}=>CBM'.format(col)] + perf1
        store.append(perf1)

store = pd.DataFrame(store, columns=['rule', 'sensitivity', 
                                     'specificity', 'accuracy', 'mcc', 'pvalue'])
store.to_csv('results_final/ml_cbm_pred/position_rules.csv')

    
    
    
    
# Correlation between tenth disulfide bond and presence of a CBM
#===================================================================#
c4 = X_val_top50['C4']
c72 = X_val_top50['C72']
c4andc72 = c4 * c72
cm = confusion_matrix(has_cbm, c4andc72)
perf = evalPerf(has_cbm, c4andc72)
#print(cm)
#print(perf)







