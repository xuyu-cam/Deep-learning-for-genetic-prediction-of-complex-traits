# Details of each script


- **multi_NN_submission_noembeds_onehot_GPU_sample_size_best_params.sh**: submit jobs to run MLP for a given list of phenotypes and a given set of one-hot encoded SNPs. There is a flag within the script to indicate the use of 10k, 50k or 100k samples for training/validating/testing.

- **multi_NN_submission_noembeds_linear_GPU_sample_size_best_params.sh**: submit jobs to run MLP for a given list of phenotypes and a given set of SNPs (using genotype dosage as input). There is a flag within the script to indicate the use of 10k, 50k or 100k samples for training/validating/testing.

- **prediction_networks_noembeds_onehot_GPU_samplesize_and_besthparams.py**: run MLP per setting (given a phenotype and sample size) with one-hot encoded genotype data, which trains nerual networks (NN) with various structures (via Talos) before selecting the best and predicting on the test set. Also it has the option to load previous best hyperparameters from previous trainings.

- **prediction_networks_noembeds_linear_GPU_samplesize_and_besthparams.py**: run MLP per setting (given a phenotype and sample size) with genotype dosage data, which trains nerual networks (NN) with various structures (via Talos) before selecting the best and predicting on the test set. Also it has the option to load previous best hyperparameters from previous trainings.



