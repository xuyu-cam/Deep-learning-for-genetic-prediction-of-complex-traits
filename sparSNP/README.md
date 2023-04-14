- SparSNP_master_sample_sizes_one_hot.sh – submitting SparSNP for a given list of phenotypes for a given set of one hot encoded genotypes. There is a flag within here to submit for 10k,50k,100k sample sizes. It calls SparSNP_sample_size_one_hot.sh with some arguments. 
- SparSNP_master_sample_sizes.sh – IDEM but not one hot encoded genotypes.  This calls SparSNP_sample_size.sh with some arguments.

- SparSNP_output_master_sample_size_one_hot.sh – used to convert the SparSNP output into a PGS for a given set of input phenotypes. It calls SparSNP_predict_final_sample_size_one_hot.sh to do this. This would be called after the jobs from (1) finished.
- SparSNP_output_master_sample_size.sh – IDEM but without one hot encoded genotypes. Used to convert the SparSNP output into a PGS for a given set of input phenotypes. It calls SparSNP_predict_final_sample_size.sh to do this. This would be called after the jobs from (2) finished.
![image](https://user-images.githubusercontent.com/61654962/232037881-fd348595-84b0-4760-88b4-281983828f10.png)
