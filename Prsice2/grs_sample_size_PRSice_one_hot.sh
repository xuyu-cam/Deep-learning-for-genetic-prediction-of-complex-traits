#!/bin/bash
#Partition: 'sysgen' (<24h) or 'sysgen_long' (>24h)
#SBATCH -p sysgen
#Give the job a name:
#SBATCH --job-name="gwas-grs"
# Maximum number of tasks/CPU cores used by the job:
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
# The amount of memory in megabytes per process in the job:
#SBATCH --mem=2g
# The maximum running time of the job in days-hours:mins:sec
#SBATCH --requeue
#SBATCH --time=0-23:0:00
# The job command(s):

#train file
file=$1
#test file
#test_file=$2
pheno=$2  # file must have ID ALLELE SCORE
output=$3
test_file=$4
p=$5

echo "train file " $file
echo "original phenotype " $pheno
echo "saving as " $output
echo "testing files" $test_file
echo "p value filtering" $p 
#exit 1
#--score my.scores(betas) (by default columns are SNP ALLELE SCORE, otherwise it can be given as parameter ex. 1 2 3)
#header if scores include a header
#By default, final scores are averages of valid per-allele scores. Otherwise use sum 
#rsid then betas
printf "\ngetting rsids and alleles\n"
awk '{print $2, $5}' $file".bim" > ~/tmp/tempfile$$

#do GWAS
printf "\nsplitting phenotype\n $pheno\n"
#~/bin/pheno_file_create.sh "$pheno"
train_size=$(wc -l ${file}".fam" | awk '{print $1}')
test_size=$(wc -l ${test_file}".fam" | awk '{print $1}')

echo $train_size
echo $test_size

awk '{print $1}' "${file}.fam" | wc -l 

#exit 1
#subsetting the phenotypes 
if [ ! -f "${pheno}_training_ss_${train_size}" ]; then
    echo "phenotype not reformatted, reformatting"
	for i in $(awk '{ORS=" "}{print $1}' "${file}.fam") ; do  grep "${i}_id2" "$pheno" ; done > "${pheno}_training_ss_${train_size}"
	~/bin/single_pheno_reshape.sh "${pheno}_training_ss_${train_size}"
fi

if [ ! -f "${pheno}_testing_ss_${test_size}" ]; then
	echo "phenotype not reformatted, reformatting"
	for i in $(awk '{ORS=" "}{print $1}' "${test_file}.fam") ; do  grep "${i}_id2" "$pheno" ; done > "${pheno}_testing_ss_${test_size}"
	~/bin/single_pheno_reshape.sh "${pheno}_testing_ss_${test_size}"
fi
##now in format we have "${pheno}_training_ss_${train_size}_reshape" "${pheno}_testing_ss_${test_size}_reshape"

echo "training file"
head "${pheno}_training_ss_${train_size}_reshape" 
echo "testing file"
head "${pheno}_testing_ss_${test_size}_reshape"

printf "\nGWAS\n"
plink1.9 --bfile $file --assoc --output-min-p 1e-307 --ci 0.95 --pheno "$pheno""_training_ss_${train_size}_reshape"  --allow-no-sex --out $output"_gwas" 
#grabbings betas
printf "\npvalue filtering\n"

#printf "\n\n\n\n\n$p\n\n\n\n\n"
awk -v var=$p ' {if ($9+0 <= var+0.0) print $0; else if ($9 =="P") print $0}' $output"_gwas.qassoc" > $output"_gwas.qassoc_filtered_"$p 

printf "\n\ncontents of filtered file\n\n"
head $output"_gwas.qassoc_filtered_"$p

printf "\n\ngrabbing betas from "$output"_gwas.qassoc\n"
awk '{print $5}' $output"_gwas.qassoc_filtered_$p"  | tail -n +2 > ~/tmp/tempfile_beta$$

printf "\ncreating scorefile as score_file\n"
paste -d " " ~/tmp/tempfile$$ ~/tmp/tempfile_beta$$ > $output"_score_file"



rm ~/tmp/tempfile$$
rm ~/tmp/tempfile_beta$$
 
#grab betas
#trimming score file from those that didnt pass the pval filter
awk  '$3!=""' $output"_score_file" > $output"_score_file_"$p
#Neff_500_nepi_500_herit_0.5_epilvl_0.2_fixedallepi_28_may_19/grs_construction_05_06_19_score_file
#do score

#awk '{}'
#/projects/sysgen/users/jgrealey/Simulations/hun_k_samples/training_dataset_final_filtered_60k.bim
#printf "\ncreating training score\n"
#plink1.9 --bfile $file --score $output"_score_file_"$p sum --allow-no-sex -out $output"_grs_$p"

#printf "\n calculating test score\n"
#plink1.9 --bfile $test_file --score $output"_score_file_"$p sum --allow-no-sex -out $output"_testgrs_$p"

#printf "\nformatting for python\n"
#awk '{print $1,$2,$3,$4,$5,$6}' "$output""_testgrs_$p.profile" > "$output""_testgrs.profile_formatted_$p"

echo  "${file}.bim" "${output}_gwas.qassoc" 
head  $file".bim" 
#head  ${output}"_gwas.qassoc"  

tail -n +2 "${output}_gwas.qassoc" |  sort -k 2  > "${output}_gwas_sorted.qassoc" 

if [ ! -f "/sysgen/workspace/users/jgrealey/jgrealey/embedding/src/src/pred_nets/grs/PRSice/sorted_bim_files/${file}_sorted_${train_size}.bim" ]; then
	echo "sorting bim file ${file}.bim "
	sort -k 2 ${file}".bim" > "/sysgen/workspace/users/jgrealey/jgrealey/embedding/src/src/pred_nets/grs/PRSice/sorted_bim_files/sorted_final_filtered_file_${train_size}.bim"
fi

echo  "/sysgen/workspace/users/jgrealey/jgrealey/embedding/src/src/pred_nets/grs/PRSice/sorted_bim_files/${file}_sorted_${train_size}.bim" "${output}_gwas_sorted.qassoc" 
#head  "/sysgen/workspace/users/jgrealey/jgrealey/embedding/src/src/pred_nets/grs/PRSice/sorted_bim_files/${file}_sorted_${train_size}.bim"
head "/sysgen/workspace/users/jgrealey/jgrealey/embedding/src/src/pred_nets/grs/PRSice/sorted_bim_files/sorted_final_filtered_file_${train_size}.bim"

head  ${output}"_gwas_sorted.qassoc"  


paste -d ' ' "/sysgen/workspace/users/jgrealey/jgrealey/embedding/src/src/pred_nets/grs/PRSice/sorted_bim_files/sorted_final_filtered_file_${train_size}.bim" ${output}"_gwas_sorted.qassoc" > ${output}"_gwas_sorted_input_file"

#echo "pasted file at ${output}_gwas_sorted_input_file"

head ${output}"_gwas_sorted_input_file"

#exit 1 
prsice_dir="/sysgen/workspace/users/jgrealey/jgrealey/embedding/src/src/pred_nets/grs/PRSice"
echo "running PRSice"

Rscript ${prsice_dir}"/"PRSice.R --dir $prsice_dir --prsice $prsice_dir/PRSice_linux --stat BETA --beta --binary-target F  --base ${output}"_gwas_sorted_input_file" --target ${test_file} --pheno "${pheno}_testing_ss_${test_size}_reshape" --A1 4 --bp 3 --A2 5 --snp 1 --index --pvalue 14 --stat 10 --out "PRSice_output_ss_${train_size}_${test_size}" --print-snp
#--keep-ambig --print-snp

location=$(pwd)

#mv "PRSice.best" "PRSice.best_${test_size}"
#cd /projects/sysgen/users/jgrealey/embedding/src/src/pred_nets/grs
pwd
#python grs_predict.py test/testing_Neff_500_nepi_500_herit_0.5_epilvl_0.6_fixedallepi_28_may_19_testgrs.profile_formatted /projects/sysgen/users/jgrealey/Simulations/hun_k_samples/src/Neff_500_nepi_500_herit_0.5_epilvl_0.6_fixedallepi_28_may_19/phenotype_numeff500numepi500repeat0.csv_sparsnp_testing
printf "\nrunning grs_predict_prsice.py with \nGRS - \n"$output"_testgrs.profile_formatted_$p\npheno as \n${pheno}_testing_ss_${test_size}_reshape\n"
#prsice_dir="/sysgen/workspace/users/jgrealey/jgrealey/embedding/src/src/pred_nets/grs"
python /sysgen/workspace/users/jgrealey/jgrealey/embedding/src/src/pred_nets/grs/PRSice/grs_predict_prsice.py "PRSice_output_ss_${train_size}_${test_size}.best" "${pheno}_testing_ss_${test_size}_reshape"

#python grs_predict.py $location"/""$output""_testgrs.profile_formatted_$p" "${pheno}_testing_ss_${test_size}_reshape"

