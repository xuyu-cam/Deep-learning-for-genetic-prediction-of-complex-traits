# Details of each script

- **SparSNP_master_sample_sizes_one_hot.sh**: submit jobs to run SparSNP for a given list of phenotypes and a given set of one-hot encoded SNPs. There is a flag within the script to indicate the use of 10k, 50k or 100k samples for training/testing. 

- **SparSNP_sample_size_one_hot.sh**: run SparSNP per setting with one-hot encoded genotype data.

- **SparSNP_master_sample_sizes.sh**: submit jobs to run SparSNP for a given list of phenotypes and a given set of SNPs (regular genotype dosage input). There is a flag within the script to indicate the use of 10k, 50k or 100k samples for training/testing. 

- **SparSNP_sample_size.sh**: run SparSNP per setting with genotype dosage data.

- **SparSNP_output_master_sample_size_one_hot.sh**: calculate PGS using SparSNP-trained model output (with one-hot encoded SNPs) for a given set of phenotypes on test sets. There is a flag within the script to indicate it was for 10k, 50k or 100k training/testing samples. 

- **SparSNP_predict_final_sample_size_one_hot.sh**: calculate PGS for a given setting on the test sample set using one-hot encoded genotype data.

- **SparSNP_output_master_sample_size.sh**: calculate PGS using SparSNP-trained model output (with SNPs dosage as input) for a given set of phenotypes on test sets. There is a flag within the script to indicate it was for 10k, 50k or 100k training/testing samples. 

- **SparSNP_predict_final_sample_size.sh**: calculate PGS for a given setting on the test sample set using genotype dosage data.

- **grs_predict_sparsnp.py**: compare SparSNP model-based PGSs with phenotype values for performance estimate.

