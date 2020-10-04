# Cel7_ML
## Using machine learning to uncover sequence-function relationships in a cellulase family
----------------

Family 7 cellulases (Cel7s), or glycoside hydrolases (GH7s),  are principal enzymes for cellulose degradation, both in nature and in industry. In this work, machine learning (ML)is applied to relate the amino acid sequence of GH7s to function by identifying key sequence features utilized by the ML algorithms that correlate with functional subtypes.

The strategies utilized in this work may be adapted to uncover sequence-function relationships in other protein families.


## Requirements
-----------------
- Python (>=3)

##### Python modules (version used in this work)
- pandas (0.24.2)
- numpy (1.16.2)
- scipy (1.1.0)
- biopython (1.73)
- scikit-learn (0.20.3)
- imbalanced-learn (0.4.3)
- matplotlib (3.0.2)
- seaborn (0.9.0)
- pydot_ng (2.0.0)

##### Other requirements (install on your machine, and specify executable path in required scripts)
- HMMER [Download __[here](http://hmmer.org/download.html)__]
- MAFFT [Download __[here](https://mafft.cbrc.jp/alignment/software/)__]
- NCBI command-line BLAST  [Download __[here](https://blast.ncbi.nlm.nih.gov/Blast.cgi?PAGE_TYPE=BlastDocs&DOC_TYPE=Download)__]


## Python scripts
-----------------------
##### Main scripts (in chronological order as used in the study)
- `subtype_hmm.py` : Use hidden Markov models (HMM) to discriminate GH7 functional subtypes (CBH vs EG)
- `subtype_ml.py`: Use supervised machine learning to discriminate GH7 functional subtypes
- `subtype_rules.py`: Derive position-specific classification rules for discriminating GH7 functional subtypes
- `cbm_ml.py`: Supervised ML to predict the presence of carbohydrate binding modules (CBM) in GH7s.

##### Other Python scripts
- `bioinformatics.py`: contains adhoc functions for bioinformatic analysis
- `plots_and_analysis.py`: for analyzing results and plotting the figures in the manuscript 


## Datasets and plots
-------------------------
- Sequence datasets are in `fasta/`
- Sequences split into five folds used for validation and design of the HMM, as well as the final trained HMMs, are in `hmm_train_test/` 
- Datasets containing results presented in the paper (Gado *et al*, 2019) are in `results_final/`
- Figures and tables in the manuscript are in `plots/`

## Citation
-----------------------
If you find this work useful, please cite this paper:

Gado, J.E., Harrison, B.E., Sandgren, M., Ståhlberg, J., Beckham, G.T., and Payne, C.M. **Machine learning reveals sequence-function relationships in family 7 glycoside hydrolases.** Submitted to *FEBS* (2020).
