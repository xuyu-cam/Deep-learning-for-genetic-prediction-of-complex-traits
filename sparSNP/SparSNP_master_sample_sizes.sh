#!/bin/bash
# declare an array called array and define 3 vales
full_fixed_sims_same_dist=('Neff_500_nepi_500_herit_0.2_epilvl_0.2_m16_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.2_RR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.2_T_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.2_XOR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.4_m16_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.4_RR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.4_T_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.4_XOR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.6_m16_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.6_RR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.6_T_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.6_XOR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.8_m16_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.8_RR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.8_T_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.8_XOR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0_m16_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0_RR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0_T_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0_XOR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0.2_m16_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0.2_RR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0.2_T_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0.2_XOR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0.4_m16_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0.4_RR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0.4_T_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0.4_XOR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0.6_m16_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0.6_RR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0.6_T_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0.6_XOR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0.8_m16_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0.8_RR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0.8_T_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0.8_XOR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0_m16_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0_RR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0_T_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0_XOR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.8_epilvl_0.2_m16_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.8_epilvl_0.2_RR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.8_epilvl_0.2_T_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.8_epilvl_0.2_XOR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.8_epilvl_0.4_m16_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.8_epilvl_0.4_RR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.8_epilvl_0.4_T_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.8_epilvl_0.4_XOR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.8_epilvl_0.6_m16_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.8_epilvl_0.6_RR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.8_epilvl_0.6_T_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.8_epilvl_0.6_XOR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.8_epilvl_0.8_m16_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.8_epilvl_0.8_RR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.8_epilvl_0.8_T_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.8_epilvl_0.8_XOR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.8_epilvl_0_m16_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.8_epilvl_0_RR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.8_epilvl_0_T_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.8_epilvl_0_XOR_model_12_Sep_19_diffseed_final')

noepidir=('Neff_500_nepi_500_herit_0.5_epilvl_0_fixed_fit_28_may_19')
RRmodel=('Neff_500_nepi_500_herit_0.5_epilvl_0.2_RRmodel_23_Jun_19' 'Neff_500_nepi_500_herit_0.5_epilvl_0.4_RRmodel_23_Jun_19' 'Neff_500_nepi_500_herit_0.5_epilvl_0.6_RRmodel_23_Jun_19' 'Neff_500_nepi_500_herit_0.5_epilvl_0.8_RRmodel_23_Jun_19' 'Neff_500_nepi_500_herit_0.5_epilvl_0_RRmodel_23_Jun_19' 'Neff_500_nepi_500_herit_0.5_epilvl_1_RRmodel_23_Jun_19' 'Neff_500_nepi_500_herit_0.8_epilvl_0.2_RRmodel_23_Jun_19' 'Neff_500_nepi_500_herit_0.8_epilvl_0.4_RRmodel_23_Jun_19' 'Neff_500_nepi_500_herit_0.8_epilvl_0.6_RRmodel_23_Jun_19' 'Neff_500_nepi_500_herit_0.8_epilvl_0.8_RRmodel_23_Jun_19' 'Neff_500_nepi_500_herit_0.8_epilvl_0_RRmodel_23_Jun_19' 'Neff_500_nepi_500_herit_0.8_epilvl_1_RRmodel_23_Jun_19')
simdir='/projects/sysgen/users/jgrealey/Simulations/hun_k_samples/src/finalised_sep_19/'
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
#spardir='/projects/sysgen/users/jgrealey/embedding/src/src/pred_nets/sparsnp'

#spardir='/projects/sysgen/users/jgrealey/embedding/src/src/pred_nets/sparsnp/ld_09_filtered_sparsnp'
#cd $spardir
#/projects/sysgen/users/jgrealey/Simulations/hun_k_samples/src/standardisation_full_genotypes_23_02_19
#jobflags="-p sysgen_long --job-name="DAE" --ntasks=1 --cpus-per-task=20 --time=13-10:0:00 "
#echo  "hello" $jobflags
#savestring='_patience9_29_may_19_fixedherit_trainableembeds'
#trainableembeds
saves=$((0))
#teststring=sleep 50s



#ss="10k"
ss="50k"
if [ "$ss" == "10k" ]; then
        genofile="/projects/sysgen/users/jgrealey/Simulations/hun_k_samples/final_data/sample_sizes/final_filtered_6k_train_samples"
        testfile="/projects/sysgen/users/jgrealey/Simulations/hun_k_samples/final_data/sample_sizes/final_filtered_4k_test_samples"
		spardir="/projects/sysgen/users/jgrealey/embedding/src/src/pred_nets/sparsnp/10k"
fi

if [ "$ss" == "50k" ]; then
        genofile="/projects/sysgen/users/jgrealey/Simulations/hun_k_samples/final_data/sample_sizes/final_filtered_30k_train_samples"
        testfile="/projects/sysgen/users/jgrealey/Simulations/hun_k_samples/final_data/sample_sizes/final_filtered_20k_test_samples"
		spardir="/projects/sysgen/users/jgrealey/embedding/src/src/pred_nets/sparsnp/50k"

fi
echo ${ss}

cd $spardir

single_test=('Neff_500_nepi_500_herit_0.8_epilvl_0_T_model_12_Sep_19_diffseed_final')
savestring="_21_may_20_SparSNP_${ss}"
for i in "${full_fixed_sims_same_dist[@]}"
#"$smalldirnew[@]}"
do
	mkdir $i
	cd $spardir"/"$i
	export PATH=/software/SparSNP/:$PATH
	#phendir=""$simdir"/"$i"/phenotype_numeff500numepi500repeat0.csv"
	fullsimdir=$simdir$i
	phendir=$(find $fullsimdir -type f -name "*phenotype_num*.csv")
	#phendir=$(find $full)
	printf "\nphenotype file\n"
	echo $phendir #"$simdir"/"$i"/phenotype_numeff500numepi500repeat0.csv	
	#genofile="/projects/sysgen/users/jgrealey/Simulations/hun_k_samples/training_dataset_final_filtered_60k"
	#genofile="/projects/sysgen/users/jgrealey/Simulations/hun_k_samples/final_data/ld_filtering/ld_filtered_final_100k_r0.9_1000wind_50step"
	numprocs=10
	maxsnps=10000
	l2=0.2
	l1min=0.001
	nfolds=5
	#inputs="/projects/sysgen/users/jgrealey/Simulations/hun_k_samples/training_dataset_filtered_60k 12 10000 0.2 "$phendir""#echo "submitted here" #"/projects/sysgen/users/jgrealey/Simulations/hun_k_samples/src/$i"
	jid1=$(sbatch --output="${ss}_sparsnp_$savestring.out" --parsable ~/scripts/src/SparSNP_sample_size.sh $genofile $numprocs $maxsnps $l1min $l2 $nfolds $phendir)
	cd $spardir
	#sleep 1
saves=$((saves+1))
done

