#!/bin/bash
# declare an array called array and define 3 vales
fixed_sims=('Neff_500_nepi_500_herit_0.5_epilvl_0.2_fixedallepi_28_may_19' 'Neff_500_nepi_500_herit_0.5_epilvl_0.4_fixedallepi_28_may_19' 'Neff_500_nepi_500_herit_0.5_epilvl_0.6_fixedallepi_28_may_19' 'Neff_500_nepi_500_herit_0.5_epilvl_0.8_fixedallepi_28_may_19' 'Neff_500_nepi_500_herit_0.5_epilvl_0_fixedallepi_28_may_19' 'Neff_500_nepi_500_herit_0.5_epilvl_1_fixedallepi_28_may_19' 'Neff_500_nepi_500_herit_0.8_epilvl_0.2_fixedallepi_28_may_19' 'Neff_500_nepi_500_herit_0.8_epilvl_0.4_fixedallepi_28_may_19' 'Neff_500_nepi_500_herit_0.8_epilvl_0.6_fixedallepi_28_may_19' 'Neff_500_nepi_500_herit_0.8_epilvl_0.8_fixedallepi_28_may_19' 'Neff_500_nepi_500_herit_0.8_epilvl_0_fixedallepi_28_may_19' 'Neff_500_nepi_500_herit_0.8_epilvl_1_fixedallepi_28_may_19')
noepidir=('Neff_500_nepi_500_herit_0.5_epilvl_0_fixed_fit_28_may_19')
RRmodel=('Neff_500_nepi_500_herit_0.5_epilvl_0.2_RRmodel_23_Jun_19' 'Neff_500_nepi_500_herit_0.5_epilvl_0.4_RRmodel_23_Jun_19' 'Neff_500_nepi_500_herit_0.5_epilvl_0.6_RRmodel_23_Jun_19' 'Neff_500_nepi_500_herit_0.5_epilvl_0.8_RRmodel_23_Jun_19' 'Neff_500_nepi_500_herit_0.5_epilvl_0_RRmodel_23_Jun_19' 'Neff_500_nepi_500_herit_0.5_epilvl_1_RRmodel_23_Jun_19' 'Neff_500_nepi_500_herit_0.8_epilvl_0.2_RRmodel_23_Jun_19' 'Neff_500_nepi_500_herit_0.8_epilvl_0.4_RRmodel_23_Jun_19' 'Neff_500_nepi_500_herit_0.8_epilvl_0.6_RRmodel_23_Jun_19' 'Neff_500_nepi_500_herit_0.8_epilvl_0.8_RRmodel_23_Jun_19' 'Neff_500_nepi_500_herit_0.8_epilvl_0_RRmodel_23_Jun_19' 'Neff_500_nepi_500_herit_0.8_epilvl_1_RRmodel_23_Jun_19')
simdir='/projects/sysgen/users/jgrealey/Simulations/hun_k_samples/src'
#m16sims=('Neff_500_nepi_500_herit_0.5_epilvl_0.2_m16model_8_Aug_19' 'Neff_500_nepi_500_herit_0.5_epilvl_0.4_m16model_8_Aug_19' 'Neff_500_nepi_500_herit_0.5_epilvl_0.6_m16model_8_Aug_19' 'Neff_500_nepi_500_herit_0.5_epilvl_0.8_m16model_8_Aug_19' 'Neff_500_nepi_500_herit_0.5_epilvl_0_m16model_8_Aug_19' 'Neff_500_nepi_500_herit_0.8_epilvl_0.2_m16model_8_Aug_19' 'Neff_500_nepi_500_herit_0.8_epilvl_0.4_m16model_8_Aug_19' 'Neff_500_nepi_500_herit_0.8_epilvl_0.6_m16model_8_Aug_19' 'Neff_500_nepi_500_herit_0.8_epilvl_0.8_m16model_8_Aug_19' 'Neff_500_nepi_500_herit_0.8_epilvl_0_m16model_8_Aug_19')
t_model=('Neff_500_nepi_500_herit_0.2_epilvl_0.2_Tmodel_13_Aug_19_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.4_Tmodel_13_Aug_19_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.6_Tmodel_13_Aug_19_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.8_Tmodel_13_Aug_19_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0_Tmodel_13_Aug_19_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0.2_Tmodel_13_Aug_19_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0.4_Tmodel_13_Aug_19_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0.6_Tmodel_13_Aug_19_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0.8_Tmodel_13_Aug_19_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0_Tmodel_13_Aug_19_final' 'Neff_500_nepi_500_herit_0.8_epilvl_0.2_Tmodel_13_Aug_19_final' 'Neff_500_nepi_500_herit_0.8_epilvl_0.4_Tmodel_13_Aug_19_final' 'Neff_500_nepi_500_herit_0.8_epilvl_0.6_Tmodel_13_Aug_19_final' 'Neff_500_nepi_500_herit_0.8_epilvl_0.8_Tmodel_13_Aug_19_final' 'Neff_500_nepi_500_herit_0.8_epilvl_0_Tmodel_13_Aug_19_final')

RR_sims=('Neff_500_nepi_500_herit_0.5_epilvl_0.2_RRmodel_23_Jun_19' 'Neff_500_nepi_500_herit_0.5_epilvl_0.4_RRmodel_23_Jun_19' 'Neff_500_nepi_500_herit_0.5_epilvl_0.6_RRmodel_23_Jun_19' 'Neff_500_nepi_500_herit_0.5_epilvl_0.8_RRmodel_23_Jun_19' 'Neff_500_nepi_500_herit_0.5_epilvl_0_RRmodel_23_Jun_19' 'Neff_500_nepi_500_herit_0.8_epilvl_0.2_RRmodel_23_Jun_19' 'Neff_500_nepi_500_herit_0.8_epilvl_0.4_RRmodel_23_Jun_19' 'Neff_500_nepi_500_herit_0.8_epilvl_0.6_RRmodel_23_Jun_19' 'Neff_500_nepi_500_herit_0.8_epilvl_0.8_RRmodel_23_Jun_19' 'Neff_500_nepi_500_herit_0.8_epilvl_0_RRmodel_23_Jun_19' 'Neff_500_nepi_500_herit_0.2_epilvl_0.2_RRmodel_13_Aug_19_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.4_RRmodel_13_Aug_19_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.6_RRmodel_13_Aug_19_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.8_RRmodel_13_Aug_19_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0_RRmodel_13_Aug_19_final')

m16_sims=('Neff_500_nepi_500_herit_0.2_epilvl_0.2_m16model_13_Aug_19_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.4_m16model_13_Aug_19_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.6_m16model_13_Aug_19_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.8_m16model_13_Aug_19_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0_m16model_13_Aug_19_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0.2_m16model_8_Aug_19' 'Neff_500_nepi_500_herit_0.5_epilvl_0.4_m16model_8_Aug_19' 'Neff_500_nepi_500_herit_0.5_epilvl_0.6_m16model_8_Aug_19' 'Neff_500_nepi_500_herit_0.5_epilvl_0.8_m16model_8_Aug_19' 'Neff_500_nepi_500_herit_0.5_epilvl_0_m16model_8_Aug_19' 'Neff_500_nepi_500_herit_0.8_epilvl_0.2_m16model_8_Aug_19' 'Neff_500_nepi_500_herit_0.8_epilvl_0.4_m16model_8_Aug_19' 'Neff_500_nepi_500_herit_0.8_epilvl_0.6_m16model_8_Aug_19' 'Neff_500_nepi_500_herit_0.8_epilvl_0.8_m16model_8_Aug_19' 'Neff_500_nepi_500_herit_0.8_epilvl_0_m16model_8_Aug_19')
embed='full_batches_tanhtest_fn5000_em2500_hdf5_04_apr_19_transposed_train_split_genotypes_standardised_full_transpose_compressed.h5_transposed_validation_split_genotypes_standardised_full_transpose_compressed.h5.npy' 
#levels of epistasis
full_fixed_sims_same_dist=('Neff_500_nepi_500_herit_0.2_epilvl_0.2_m16_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.2_RR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.2_T_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.2_XOR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.4_m16_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.4_RR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.4_T_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.4_XOR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.6_m16_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.6_RR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.6_T_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.6_XOR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.8_m16_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.8_RR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.8_T_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.8_XOR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0_m16_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0_RR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0_T_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0_XOR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0.2_m16_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0.2_RR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0.2_T_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0.2_XOR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0.4_m16_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0.4_RR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0.4_T_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0.4_XOR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0.6_m16_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0.6_RR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0.6_T_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0.6_XOR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0.8_m16_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0.8_RR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0.8_T_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0.8_XOR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0_m16_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0_RR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0_T_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0_XOR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.8_epilvl_0.2_m16_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.8_epilvl_0.2_RR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.8_epilvl_0.2_T_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.8_epilvl_0.2_XOR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.8_epilvl_0.4_m16_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.8_epilvl_0.4_RR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.8_epilvl_0.4_T_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.8_epilvl_0.4_XOR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.8_epilvl_0.6_m16_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.8_epilvl_0.6_RR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.8_epilvl_0.6_T_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.8_epilvl_0.6_XOR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.8_epilvl_0.8_m16_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.8_epilvl_0.8_RR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.8_epilvl_0.8_T_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.8_epilvl_0.8_XOR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.8_epilvl_0_m16_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.8_epilvl_0_RR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.8_epilvl_0_T_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.8_epilvl_0_XOR_model_12_Sep_19_diffseed_final')



ss="100k"
#ss="50k"
#ss="10k"

if [ "$ss" == "100k" ]; then 
	genofile="/projects/sysgen/users/jgrealey/Simulations/hun_k_samples/training_dataset_final_filtered_60k"
	testfile="/projects/sysgen/users/jgrealey/Simulations/hun_k_samples/testing_dataset_final_filtered_40k"
fi

if [ "$ss" == "10k" ]; then 
	genofile="/projects/sysgen/users/jgrealey/Simulations/hun_k_samples/final_data/sample_sizes/final_filtered_6k_train_samples"
	testfile="/projects/sysgen/users/jgrealey/Simulations/hun_k_samples/final_data/sample_sizes/final_filtered_4k_test_samples"
fi

if [ "$ss" == "50k" ]; then 
	genofile="/projects/sysgen/users/jgrealey/Simulations/hun_k_samples/final_data/sample_sizes/final_filtered_30k_train_samples"
	testfile="/projects/sysgen/users/jgrealey/Simulations/hun_k_samples/final_data/sample_sizes/final_filtered_20k_test_samples"

fi
echo ${ss}

re_run_50k=('Neff_500_nepi_500_herit_0.2_epilvl_0.2_T_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.4_T_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0_m16_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.4_m16_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.4_RR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.8_RR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.4_XOR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.8_XOR_model_12_Sep_19_diffseed_final' )
re_run_100k=('Neff_500_nepi_500_herit_0.2_epilvl_0.4_T_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.4_XOR_model_12_Sep_19_diffseed_final')
re_run_50k2=('Neff_500_nepi_500_herit_0.2_epilvl_0.4_m16_model_12_Sep_19_diffseed_final')
re_run_10k=('Neff_500_nepi_500_herit_0.5_epilvl_0.8_RR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.2_RR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.6_RR_model_12_Sep_19_diffseed_final') 
re_run=('Neff_500_nepi_500_herit_0.8_epilvl_0.8_XOR_model_12_Sep_19_diffseed_final')
#turning on or off main effects
grsdir='/projects/sysgen/users/jgrealey/embedding/src/src/pred_nets/grs/PRSice'
cd $grsdir
savestring="PRSice_grs_construction_2nd_Jul_2020_sample_size_showsnp_${ss}"
#trainableembeds
saves=$((0))
#pval="1e4"

#sort -k 2 ${genofile} > "/projects/sysgen/users/jgrealey/embedding/src/src/pred_nets/grs/PRSice/sorted_bim_files/sorted_bim_file_${ss}.bim"
simdir="/projects/sysgen/users/jgrealey/Simulations/hun_k_samples/src/finalised_sep_19/"
#pheno_file=$(find  -type f -name "*phenotype_num*")
#teststring=sleep 50s

single_test=('Neff_500_nepi_500_herit_0.8_epilvl_0_T_model_12_Sep_19_diffseed_final')
for i in "${re_run_100k[@]}"
#for i in "${single_test[@]}"
#for i in "${full_fixed_sims_same_dist[@]}"
#"$smalldirnew[@]}"
do
	mkdir $i
	cd $grsdir"/"$i
	#echo "exporting path"
	fullsimdir=$simdir$i
	phendir=$(find $fullsimdir -type f -name "*phenotype_num*.csv")
	#phendir=""$simdir"/"$i"/phenotype_numeff500numepi500repeat0.csv"
	#echo "currently at"
	#pwd
	printf "\nphenotype file\n"
	echo $phendir #"$simdir"/"$i"/phenotype_numeff500numepi500repeat0.csv	
	##genofile="/projects/sysgen/users/jgrealey/Simulations/hun_k_samples/training_dataset_final_filtered_60k"
	#printf "using genofile with perfectly filtered variants"
	pwd
	#genofile="/projects/sysgen/users/jgrealey/Simulations/hun_k_samples/final_data/ld_filtering/ld_filtered_final_100k_r0.9_1000wind_50step"
	#ld_filtered_final_100k_ralmost1_1000wind_50step
	#testfile="/projects/sysgen/users/jgrealey/Simulations/hun_k_samples/testing_dataset_final_filtered_40k"
	pval="1"
	echo ${genofile} ${phendir} ${savestring} ${testfile} ${pval}
	#echo $i"/slurm-"%j"_$pval_${ss}.out" 
	#jid1=$(sbatch --parsable --output=$i"/slurm-"%j"_$pval_${ss}.out" ~/scripts/src/grs_sample_size.sh ${genofile} ${phendir} ${i}"/"${savestring} ${testfile} ${pval})
	sbatch --output "${ss}_grs_$savestring.out" ~/scripts/src/grs_sample_size_PRSice.sh ${genofile} ${phendir} ${savestring} ${testfile} ${pval}
	#hello here
	#printf "\n"# back to sparsnp\n
	cd $grsdir
	#pwd
	#echo "back at grssnp"


	#sleep 0.1
saves=$((saves+1))
done
