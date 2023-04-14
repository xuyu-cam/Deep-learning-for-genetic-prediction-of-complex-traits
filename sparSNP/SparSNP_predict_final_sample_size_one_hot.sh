#!/bin/bash
#Partition: 'sysgen' (<24h) or 'sysgen_long' (>24h)
#SBATCH -p sysgen
#Give the job a name:
#SBATCH --job-name="sparpred"
# Maximum number of tasks/CPU cores used by the job:
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
# The amount of memory in megabytes per process in the job (try to put the smaller file first):
#SBATCH --mem=20048
# The maximum running time of the job in days-hours:mins:sec
#SBATCH --time=0-23:0:00

#test set plink file
testset=$1

#true unshaped phenotype
pheno=$2
echo $testset
echo $pheno

test_size=$(wc -l ${testset}".fam" | awk '{print $1}')
echo $test_size

#this script calculates GRS using SPARSNP
#then uses linear regression to predict the test phenotype
#prior to this /software/SparSNP/eval.R must be run to obtain ideal number of SNPs
echo "reformatting phenotype"
for i in $(awk '{ORS=" "}{print $1}' "${testset}.fam") ; do  grep "${i}_id2" "$pheno" ; done > "${pheno}_testing_ss_oh_${test_size}"
        ~/bin/single_pheno_reshape.sh "${pheno}_testing_ss_oh_${test_size}"

echo "head oh phenotype file"
head "$pheno""_testing_ss_oh_${test_size}_reshape"

printf "\nfirst get outputs\n"

Rscript /software/SparSNP/eval.R  >output_eval.txt

#this grabs number of SNPs from output file
numSNPs=$(tail -n 1 output_eval.txt | awk -F ' ' '{print $4}')

printf "\n getting models\n"

printf "\nnumsnps : $numSNPs \n"
#for i in Neff_500_nepi_500_herit_0.* ; do printf $i; cd $i; Rscript /software/SparSNP/eval.R > output_eval.txt ; cd /projects/sysgen/users/jgrealey/embedding/src/src/pred_nets/sparsnp/ ; done
Rscript /software/SparSNP/getmodels.R nzreq=$numSNPs

printf "\nGRS creation\n"

#sbatch /home/rcanovas/scripts/plink/grs.sh /projects/sysgen/users/jgrealey/Simulations/hun_k_samples/testing_dataset_final_filtered_40k /projects/sysgen/users/jgrealey/embedding/src/src/pred_nets/sparsnp/oldresults/firsttest/discovery/avg_weights_opt.score GRS_predictioni


#sbatch /home/rcanovas/scripts/plink/grs.sh /projects/sysgen/users/jgrealey/Simulations/hun_k_samples/testing_dataset_final_filtered_40k /projects/sysgen/users/jgrealey/embedding/src/src/pred_nets/sparsnp/oldresults/firsttest/discovery/avg_weights_opt.score GRS_prediction
#plink1.9 --bfile /projects/sysgen/users/jgrealey/Simulations/hun_k_samples/testing_dataset_final_filtered_40k --score discovery/avg_weights_opt.score sum -out GRS_prediction_sparsnp

plink1.9 --bfile $testset --score discovery/avg_weights_opt.score sum -out GRS_prediction_sparsnp_one_hot_${test_size}


#next with weights, calculated GRS

printf "\nGRS reformatting\n"

~/scripts/src/reshape_plink_grs.sh GRS_prediction_sparsnp_one_hot_${test_size}.profile

printf "\npython regression\n"
python /sysgen/workspace/users/jgrealey/jgrealey/embedding/src/src/pred_nets/sparsnp/one_hot/grs_predict_sparsnp_oh.py  GRS_prediction_sparsnp_one_hot_${test_size}.profile_formatted "$pheno""_testing_ss_oh_${test_size}_reshape"

