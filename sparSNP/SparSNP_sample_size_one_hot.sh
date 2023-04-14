#!/bin/bash
#Partition: 'sysgen_long' (<24h) or 'sysgen_long' (>24h)
#SBATCH -p sysgen_long
#Give the job a name:
#SBATCH --job-name="SparSNP"

# Maximum number of tasks/CPU cores used by the job:
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#excluding node 6 
#SBATCH --exclude=bhri-hpcn-06
# The amount of memory in megabytes per process in the job (try to put the smaller file first):
#SBATCH --mem=35G
#SBATCH --requeue
# The maximum running time of the job in days-hours:mins:sec
#SBATCH --time=13-20:0:00

# The job command(s):

pfile=$1  #plink file prefix(.bim/.bed/.fam)
nprocs=$2  #number of processors
nzmax=$3   #number number of SNPs allowed in the model
l1min=$4   #
l2=$5
nfolds=$6
pheno=$7 #location of phenotype file not split
#ss=$8
echo "plink" $pfile
echo "number processes" $nprocs
echo "number snps max" $nzmax
echo "min l1" $l1min
echo "l2 penalisation" $l2
echo "number of folds" $nfolds
echo $pheno

#spliting IDs into sparSNP format # IID FID pheno
#~/bin/pheno_file_create.sh "$pheno"

train_size=$(wc -l ${pfile}".fam" | awk '{print $1}')
#test_size=$(wc -l ${test_file}".fam" | awk '{print $1}')

echo $train_size
#echo $test_size

#if [ ! -f "${pheno}_training_ss_${train_size}" ]; then
    echo "phenotype not reformatted, reformatting"
        for i in $(awk '{ORS=" "}{print $1}' "${pfile}.fam") ; do  grep "${i}_id2" "$pheno" ; done > "${pheno}_training_ss_${train_size}"
        ~/bin/single_pheno_reshape.sh "${pheno}_training_ss_${train_size}"
#fi	

#head "$pheno""_training_ss_${train_size}_reshape"

phenoin="$pheno""_training_ss_${train_size}_reshape"
echo $phenoin
head $phenoin

NUMPROCS=$nprocs NZMAX=$nzmax L1MIN=$l1min PHENO=$phenoin NFOLDS=$nfolds LAMBDA2=$l2 /software/SparSNP/crossval.sh $pfile linear  2>&1 | tee log

