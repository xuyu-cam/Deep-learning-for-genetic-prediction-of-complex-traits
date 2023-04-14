#!/bin/bash
# declare an array called array and define 3 vales
full_fixed_sims_same_dist=('Neff_500_nepi_500_herit_0.2_epilvl_0.2_m16_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.2_RR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.2_T_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.2_XOR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.4_m16_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.4_RR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.4_T_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.4_XOR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.6_m16_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.6_RR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.6_T_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.6_XOR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.8_m16_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.8_RR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.8_T_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.8_XOR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0_m16_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0_RR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0_T_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0_XOR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0.2_m16_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0.2_RR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0.2_T_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0.2_XOR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0.4_m16_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0.4_RR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0.4_T_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0.4_XOR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0.6_m16_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0.6_RR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0.6_T_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0.6_XOR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0.8_m16_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0.8_RR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0.8_T_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0.8_XOR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0_m16_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0_RR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0_T_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0_XOR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.8_epilvl_0.2_m16_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.8_epilvl_0.2_RR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.8_epilvl_0.2_T_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.8_epilvl_0.2_XOR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.8_epilvl_0.4_m16_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.8_epilvl_0.4_RR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.8_epilvl_0.4_T_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.8_epilvl_0.4_XOR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.8_epilvl_0.6_m16_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.8_epilvl_0.6_RR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.8_epilvl_0.6_T_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.8_epilvl_0.6_XOR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.8_epilvl_0.8_m16_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.8_epilvl_0.8_RR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.8_epilvl_0.8_T_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.8_epilvl_0.8_XOR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.8_epilvl_0_m16_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.8_epilvl_0_RR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.8_epilvl_0_T_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.8_epilvl_0_XOR_model_12_Sep_19_diffseed_final')

noepidir=('Neff_500_nepi_500_herit_0.5_epilvl_0_fixed_fit_28_may_19')
RRmodel=('Neff_500_nepi_500_herit_0.5_epilvl_0.2_RRmodel_23_Jun_19' 'Neff_500_nepi_500_herit_0.5_epilvl_0.4_RRmodel_23_Jun_19' 'Neff_500_nepi_500_herit_0.5_epilvl_0.6_RRmodel_23_Jun_19' 'Neff_500_nepi_500_herit_0.5_epilvl_0.8_RRmodel_23_Jun_19' 'Neff_500_nepi_500_herit_0.5_epilvl_0_RRmodel_23_Jun_19' 'Neff_500_nepi_500_herit_0.5_epilvl_1_RRmodel_23_Jun_19' 'Neff_500_nepi_500_herit_0.8_epilvl_0.2_RRmodel_23_Jun_19' 'Neff_500_nepi_500_herit_0.8_epilvl_0.4_RRmodel_23_Jun_19' 'Neff_500_nepi_500_herit_0.8_epilvl_0.6_RRmodel_23_Jun_19' 'Neff_500_nepi_500_herit_0.8_epilvl_0.8_RRmodel_23_Jun_19' 'Neff_500_nepi_500_herit_0.8_epilvl_0_RRmodel_23_Jun_19' 'Neff_500_nepi_500_herit_0.8_epilvl_1_RRmodel_23_Jun_19')
simdir='/sysgen/workspace/users/jgrealey/jgrealey/Simulations/hun_k_samples/src/finalised_sep_19/'
embed='full_batches_tanhtest_fn5000_em2500_hdf5_04_apr_19_transposed_train_split_genotypes_standardised_full_transpose_compressed.h5_transposed_validation_split_genotypes_standardised_full_transpose_compressed.h5.npy' 
#levels of epistasis
faultdir=('Neff_500_nepi_500_herit_0.5_epilvl_0.6_RRmodel_23_Jun_19')
#turning on or off main effects
m16_sims_low_herit=('Neff_500_nepi_500_herit_0.5_epilvl_0.2_m16model_8_Aug_19' 'Neff_500_nepi_500_herit_0.5_epilvl_0.4_m16model_8_Aug_19' 'Neff_500_nepi_500_herit_0.5_epilvl_0.6_m16model_8_Aug_19' 'Neff_500_nepi_500_herit_0.5_epilvl_0.8_m16model_8_Aug_19' 'Neff_500_nepi_500_herit_0.5_epilvl_0_m16model_8_Aug_19' 'Neff_500_nepi_500_herit_0.2_epilvl_0.2_m16model_13_Aug_19_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.4_m16model_13_Aug_19_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.6_m16model_13_Aug_19_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.8_m16model_13_Aug_19_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0_m16model_13_Aug_19_final' )
m16_sims_high_herit=('Neff_500_nepi_500_herit_0.8_epilvl_0.2_m16model_8_Aug_19' 'Neff_500_nepi_500_herit_0.8_epilvl_0.4_m16model_8_Aug_19' 'Neff_500_nepi_500_herit_0.8_epilvl_0.6_m16model_8_Aug_19' 'Neff_500_nepi_500_herit_0.8_epilvl_0.8_m16model_8_Aug_19' 'Neff_500_nepi_500_herit_0.8_epilvl_0_m16model_8_Aug_19')
m16_sims=('Neff_500_nepi_500_herit_0.2_epilvl_0.2_m16model_13_Aug_19_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.4_m16model_13_Aug_19_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.6_m16model_13_Aug_19_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.8_m16model_13_Aug_19_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0_m16model_13_Aug_19_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0.2_m16model_8_Aug_19' 'Neff_500_nepi_500_herit_0.5_epilvl_0.4_m16model_8_Aug_19' 'Neff_500_nepi_500_herit_0.5_epilvl_0.6_m16model_8_Aug_19' 'Neff_500_nepi_500_herit_0.5_epilvl_0.8_m16model_8_Aug_19' 'Neff_500_nepi_500_herit_0.5_epilvl_0_m16model_8_Aug_19' 'Neff_500_nepi_500_herit_0.8_epilvl_0.2_m16model_8_Aug_19' 'Neff_500_nepi_500_herit_0.8_epilvl_0.4_m16model_8_Aug_19' 'Neff_500_nepi_500_herit_0.8_epilvl_0.6_m16model_8_Aug_19' 'Neff_500_nepi_500_herit_0.8_epilvl_0.8_m16model_8_Aug_19' 'Neff_500_nepi_500_herit_0.8_epilvl_0_m16model_8_Aug_19')
t_model=('Neff_500_nepi_500_herit_0.5_epilvl_0.2_Tmodel_13_Aug_19_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0.4_Tmodel_13_Aug_19_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0.6_Tmodel_13_Aug_19_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0.8_Tmodel_13_Aug_19_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0_Tmodel_13_Aug_19_final' 'Neff_500_nepi_500_herit_0.8_epilvl_0.2_Tmodel_13_Aug_19_final' 'Neff_500_nepi_500_herit_0.8_epilvl_0.4_Tmodel_13_Aug_19_final' 'Neff_500_nepi_500_herit_0.8_epilvl_0.6_Tmodel_13_Aug_19_final' 'Neff_500_nepi_500_herit_0.8_epilvl_0.8_Tmodel_13_Aug_19_final' 'Neff_500_nepi_500_herit_0.8_epilvl_0_Tmodel_13_Aug_19_final')
RR_sims=('Neff_500_nepi_500_herit_0.5_epilvl_0.2_RRmodel_23_Jun_19' 'Neff_500_nepi_500_herit_0.5_epilvl_0.4_RRmodel_23_Jun_19' 'Neff_500_nepi_500_herit_0.5_epilvl_0.6_RRmodel_23_Jun_19' 'Neff_500_nepi_500_herit_0.5_epilvl_0.8_RRmodel_23_Jun_19' 'Neff_500_nepi_500_herit_0.5_epilvl_0_RRmodel_23_Jun_19' 'Neff_500_nepi_500_herit_0.8_epilvl_0.2_RRmodel_23_Jun_19' 'Neff_500_nepi_500_herit_0.8_epilvl_0.4_RRmodel_23_Jun_19' 'Neff_500_nepi_500_herit_0.8_epilvl_0.6_RRmodel_23_Jun_19' 'Neff_500_nepi_500_herit_0.8_epilvl_0.8_RRmodel_23_Jun_19' 'Neff_500_nepi_500_herit_0.8_epilvl_0_RRmodel_23_Jun_19' 'Neff_500_nepi_500_herit_0.2_epilvl_0.2_RRmodel_13_Aug_19_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.4_RRmodel_13_Aug_19_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.6_RRmodel_13_Aug_19_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.8_RRmodel_13_Aug_19_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0_RRmodel_13_Aug_19_final')
RR_sims_low_herit=('Neff_500_nepi_500_herit_0.2_epilvl_0.2_RRmodel_13_Aug_19_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.4_RRmodel_13_Aug_19_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.6_RRmodel_13_Aug_19_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.8_RRmodel_13_Aug_19_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0_RRmodel_13_Aug_19_final')
t_model_low_herit=('Neff_500_nepi_500_herit_0.2_epilvl_0.2_Tmodel_13_Aug_19_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.4_Tmodel_13_Aug_19_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.6_Tmodel_13_Aug_19_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.8_Tmodel_13_Aug_19_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0_Tmodel_13_Aug_19_final')

ld_filt_run=('Neff_500_nepi_500_herit_0.5_epilvl_0.4_m16_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0.4_RR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0.4_T_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0.4_XOR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0_m16_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0_RR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0_T_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0_XOR_model_12_Sep_19_diffseed_final')
ld_filt_topup=('Neff_500_nepi_500_herit_0.5_epilvl_0.2_m16_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0.2_RR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0.2_T_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0.2_XOR_model_12_Sep_19_diffseed_final' )
#savestring='_patience9_29_may_19_fixedherit_trainableembeds'
saves=$((0))

# sample size
#ss="100k"
#ss="10k"
ss="50k"
if [ "$ss" == "10k" ]; then
        genofile="/sysgen/workspace/users/jgrealey/jgrealey/Simulations/hun_k_samples/final_data/sample_sizes/one_hot/final_filtered_6k_train_samples_one_hot"
        testfile="/sysgen/workspace/users/jgrealey/jgrealey/Simulations/hun_k_samples/final_data/sample_sizes/one_hot/final_filtered_4k_test_samples_one_hot"
		spardir="/sysgen/workspace/users/jgrealey/jgrealey/embedding/src/src/pred_nets/sparsnp/one_hot/10k"
fi

if [ "$ss" == "50k" ]; then
        genofile="/sysgen/workspace/users/jgrealey/jgrealey/Simulations/hun_k_samples/final_data/sample_sizes/one_hot/final_filtered_30k_train_samples_one_hot"
        testfile="/sysgen/workspace/users/jgrealey/jgrealey/Simulations/hun_k_samples/final_data/sample_sizes/one_hot/final_filtered_20k_test_samples_one_hot"
		spardir="/sysgen/workspace/users/jgrealey/jgrealey/embedding/src/src/pred_nets/sparsnp/one_hot/50k"

fi

if [ "$ss" == "100k" ]; then
        genofile="/sysgen/workspace/users/jgrealey/jgrealey/Simulations/hun_k_samples/final_data/sample_sizes/one_hot/final_filtered_60k_train_samples_one_hot"
        testfile="/sysgen/workspace/users/jgrealey/jgrealey/Simulations/hun_k_samples/final_data/sample_sizes/one_hot/final_filtered_40k_test_samples_one_hot"
		spardir="/sysgen/workspace/users/jgrealey/jgrealey/embedding/src/src/pred_nets/sparsnp/one_hot/100k"

fi

echo ${ss}

cd $spardir

single_test=('Neff_500_nepi_500_herit_0.8_epilvl_0_T_model_12_Sep_19_diffseed_final')
savestring="_21_may_20_SparSNP_${ss}"
for i in "${full_fixed_sims_same_dist[@]}"
#"$smalldirnew[@]}"
do
	
	FILE="$i"
	#echo $i
	if [ -d "$FILE" ]; then
    	echo "$FILE exists."
	else
		echo "$FILE" doesnt exist
		mkdir $i
	fi
	#mkdir $i
	cd $spardir"/"$i
	export PATH=/software/SparSNP/:$PATH
	#phendir=""$simdir"/"$i"/phenotype_numeff500numepi500repeat0.csv"
	fullsimdir=$simdir$i
	phendir=$(find $fullsimdir -type f -name "*phenotype_num*.csv")
	#phendir=$(find $full)
	#printf "\nphenotype file\n"
	#echo $phendir #"$simdir"/"$i"/phenotype_numeff500numepi500repeat0.csv	
	#genofile="/projects/sysgen/users/jgrealey/Simulations/hun_k_samples/training_dataset_final_filtered_60k"
	#genofile="/projects/sysgen/users/jgrealey/Simulations/hun_k_samples/final_data/ld_filtering/ld_filtered_final_100k_r0.9_1000wind_50step"
	numprocs=10
	maxsnps=10000
	l2=0.2
	l1min=0.001
	nfolds=5
	#inputs="/projects/sysgen/users/jgrealey/Simulations/hun_k_samples/training_dataset_filtered_60k 12 10000 0.2 "$phendir""#echo "submitted here" #"/projects/sysgen/users/jgrealey/Simulations/hun_k_samples/src/$i"
	FILE="discovery"
	if [ -d "$FILE" ]; then
		#echo $i
		echo "$FILE exists."
	else
		echo "$FILE doesn't exist, submitting"
		jid1=$(sbatch --output="${ss}_sparsnp_$savestring.out" --parsable ~/scripts/src/SparSNP_sample_size_one_hot.sh $genofile $numprocs $maxsnps $l1min $l2 $nfolds $phendir) || sleep 600 && scontrol requeue $jid1
		echo $jid1
	fi
	cd $spardir
	#sleep 1
saves=$((saves+1))
done

