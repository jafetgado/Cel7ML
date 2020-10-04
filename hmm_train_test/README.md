#### `hmm_train_test/`: Data files from five-fold cross validation of HMM method.

`ncbi_kfold/`: Contains the NCBI sequences (427) split to 5 folds for training and testing (`cbh_train1.fasta`, `egl_train1.fasta`, `cbh_test1.fasta`, `egl_test1.fasta`, etc.), the respective HMMs built on the training data (`cbh1.hmm`, `egl1.hmm`, etc.), and the output of the HMMs applied to the test data (`cbhhmm_cbh1.out`, `cbhhmm_egl1.out`, `eglhmm_cbh1.out`, `eglhmm_egl1.out`)

`swiss_kfold/`: Contains the same files as in `ncbi_kfold`, but for the SwissProt sequences (44).

`final_hmm/`: Contains the final HMM trained on all sequences (i.e. all 5 folds) - `cbh_ncbi.hmm`, `egl_ncbi.hmm`, `cbh_swiss.hmm`, and `egl_swiss.hmm`. The subdirectory, `hmm_to_1748`, contains the output files of these HMMs applied to the NCBI dataset of 1,748 sequences (`cel7_nr99_cbhncbi.out`, etc).


