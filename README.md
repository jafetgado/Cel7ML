# Cel7ML
## Using machine learning to uncover sequence-function relationships in a cellulase family


Family 7 cellulases (Cel7s), or glycoside hydrolases (GH7s),  are principal enzymes for cellulose degradation, both in nature and in industry. In this work, machine learning (ML)is applied to relate the amino acid sequence of GH7s to function by identifying what parts of the sequence are majorly utilized by ML algorithms to predict functional subtypes.

The strategies utilized here may be adapted to uncover sequence-function relationships in other protein families.


## Requirements
- Python (>= 3)

#### Python modules (version used in this work)
- Pandas (0.24.2)
- Numpy (1.16.2)
- Scipy (1.1.0)
- BioPython (1.73)
- Sklearn (0.20.3)
- Imblearn (0.4.3)
- Matplotlib (3.0.2)
- Seaborn (0.9.0)

#### Other requirements (install on your machine, and specify executable path in required scripts)
- HMMER [Download __[here](http://hmmer.org/download.html)__]
- MAFFT [Download __[here](https://mafft.cbrc.jp/alignment/software/)__]
- NCBI command-line BLAST  [Download __[here](https://blast.ncbi.nlm.nih.gov/Blast.cgi?PAGE_TYPE=BlastDocs&DOC_TYPE=Download)__]


## Python scripts
#### Main scripts
- `subtype_hmm.py` : Using hidden Markov models (HMM) to discriminate GH7 functional subtypes (CBH vs EG)
- ``subtype_ml.py``: Supervised ML to discriminate GH7 functional subtypes
- ``subtype_rules.py``: Deriving position-specific classification rules for discriminating GH7 functional subtypes
- ``cbm_ml.py``: Supervised ML to predict the presence of carbohydrate binding modules (CBM) in GH7s.

#### Other Python scripts
- `bioinformatics.py`: adhoc functions for bioinformatic analysis
- `analysis_and_plots`: script for analyzing results and plotting figures

## Datasets
- Sequence datasets are in `./fasta/`
- Datasets containing results presented in the paper (Gado *et al*, 2019) are in `./results_final/`

## Citation
If you find this work useful, please cite this paper:

Gado, J.E., Harrison, B.E., Sandgren, M., Ståhlberg, J., Beckham, G.T., and Payne, C.M. **Machine learning reveals sequence-function relationships in family 7 glycoside hydrolases.** Submitted to *FEBS* (2019).
