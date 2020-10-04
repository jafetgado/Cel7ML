#### `ml_cbm_pred/`: results of machine learning applied to predict presence of CBM in GH7 sequences


`perf_all.csv`: Performance of random forest classifier trained on all 5,933 features.

`featimp_all.csv`: Feature importances of 5,933 features.

`perf_top50.csv`: Performance of random forest classifier trained on only the top 50 features.

`featimp_top50.csv`: Feature importances of 50 features in the random forest classifier trained on only the top 50 features.

`perf_top50_noterm.csv`: Performance of random forest classifier trained on top 50 features but with features from C-terminal residues removed (i.e 6 removed, trained on 44 features).

`poistion_rules.csv`: Position-specific classification rules derived from top 50 features.

`perf_top20.csv`: Performance of random forest classifier trained on only the top 20 features.

`featimp_top20.csv`: Feature importances of 20 features in the random forest classifier trained on only the top 20 features.

`perf_test_set.csv`: Performance of random forest classifier trained on top 20 features but validated on the separate test set.

