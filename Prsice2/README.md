# Details of each script


- **grs_multi_sample_size_PRSice_one_hot.sh**: submit jobs to run PRSice2 for a given list of phenotypes and a given set of one-hot encoded SNPs. There is a flag within the script to indicate the use of 10k, 50k or 100k samples for training/testing.

- **grs_multi_sample_size_PRSice.sh**: submit jobs to run PRSice2 for a given list of phenotypes and a given set of SNPs (using genotype dosage as input). There is a flag within the script to indicate the use of 10k, 50k or 100k samples for training/testing.

-	**grs_sample_size_PRSice_one_hot.sh**: calculate PGS using PRSice2-trained model output (with one-hot encoded SNPs) for a given set of phenotypes on test sets. There is a flag within the script to indicate it was for 10k, 50k or 100k training/testing samples.

- **grs_sample_size_PRSice.sh**: calculate PGS using PRSice2-trained model output (with SNPs dosage as input) for a given set of phenotypes on test sets. There is a flag within the script to indicate it was for 10k, 50k or 100k training/testing samples.

- **grs_predict_prsice.py**: compare SparSNP model-based PGSs with phenotype values for performance estimate on test sets.

