#### `results_final/`: final results of the analysis, mostly as spreadsheets (.csv)

`swiss_kfold.csv`: HMM alignment scores of 44 SwissProt sequences, determined by 5-fold cross validation.

`ncbi_kfold.csv`: HMM alignment scores of 427 NCBI sequences, determined by a 5-fold cross validation.

`cel7_subtypes.csv`: Predicted subtypes (CBH/EG) of 1,748 GH7s using the final HMMs.

`looplength.csv`: The number of residues in each active-site loop of 1,748 GH7s.

`cbm_blast_output.csv`: Output of BLASTing TreCel7A CBM against the database of 1,748 GH7 sequences, so as to determine which sequences possess a CBM.

`has_cbm.csv`: CBM data of all 1,748 GH7s, 1 indicates presence of CBM, and 0 indicates absence of CBM.

`aromatic_residues.csv`: Data of residues within 6 Å of the cellononaose ligand in TreCel7A that are aromatic in the CBH or EG consensus sequences.

`residue_distances.csv`: Shortest distance between each residue in TreCel7A (1 to 434) and the cellonoaose ligand.

`ml_subtype_pred/`: Directory containing results of machine learning applied to discriminate CBH and EG using their loop lengths.

`rules/`: Directory containing results of position-specific classification rules.

`ml_cbm_pred/`: Directory containing results of machine learning applied to predict presence of CBM in GH7 sequences.



