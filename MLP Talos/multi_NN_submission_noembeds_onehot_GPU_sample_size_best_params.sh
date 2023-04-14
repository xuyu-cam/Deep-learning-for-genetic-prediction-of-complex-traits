#!/bin/bash
t_model=('Neff_500_nepi_500_herit_0.2_epilvl_0.2_Tmodel_13_Aug_19_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.4_Tmodel_13_Aug_19_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.6_Tmodel_13_Aug_19_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.8_Tmodel_13_Aug_19_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0_Tmodel_13_Aug_19_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0.2_Tmodel_13_Aug_19_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0.4_Tmodel_13_Aug_19_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0.6_Tmodel_13_Aug_19_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0.8_Tmodel_13_Aug_19_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0_Tmodel_13_Aug_19_final' 'Neff_500_nepi_500_herit_0.8_epilvl_0.2_Tmodel_13_Aug_19_final' 'Neff_500_nepi_500_herit_0.8_epilvl_0.4_Tmodel_13_Aug_19_final' 'Neff_500_nepi_500_herit_0.8_epilvl_0.6_Tmodel_13_Aug_19_final' 'Neff_500_nepi_500_herit_0.8_epilvl_0.8_Tmodel_13_Aug_19_final' 'Neff_500_nepi_500_herit_0.8_epilvl_0_Tmodel_13_Aug_19_final')
t_sims_low_herit=('Neff_500_nepi_500_herit_0.2_epilvl_0.2_Tmodel_13_Aug_19_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.4_Tmodel_13_Aug_19_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.6_Tmodel_13_Aug_19_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.8_Tmodel_13_Aug_19_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0_Tmodel_13_Aug_19_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0.2_Tmodel_13_Aug_19_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0.4_Tmodel_13_Aug_19_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0.6_Tmodel_13_Aug_19_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0.8_Tmodel_13_Aug_19_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0_Tmodel_13_Aug_19_final') 
t_sims_high_herit=('Neff_500_nepi_500_herit_0.8_epilvl_0.2_Tmodel_13_Aug_19_final' 'Neff_500_nepi_500_herit_0.8_epilvl_0.4_Tmodel_13_Aug_19_final' 'Neff_500_nepi_500_herit_0.8_epilvl_0.6_Tmodel_13_Aug_19_final' 'Neff_500_nepi_500_herit_0.8_epilvl_0.8_Tmodel_13_Aug_19_final' 'Neff_500_nepi_500_herit_0.8_epilvl_0_Tmodel_13_Aug_19_final')

RR_sims_high_herit=('Neff_500_nepi_500_herit_0.8_epilvl_0.2_RRmodel_23_Jun_19' 'Neff_500_nepi_500_herit_0.8_epilvl_0.4_RRmodel_23_Jun_19' 'Neff_500_nepi_500_herit_0.8_epilvl_0.6_RRmodel_23_Jun_19' 'Neff_500_nepi_500_herit_0.8_epilvl_0.8_RRmodel_23_Jun_19' 'Neff_500_nepi_500_herit_0.8_epilvl_0_RRmodel_23_Jun_19')
RR_sims_low_herit=('Neff_500_nepi_500_herit_0.2_epilvl_0.2_RRmodel_13_Aug_19_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.4_RRmodel_13_Aug_19_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.6_RRmodel_13_Aug_19_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.8_RRmodel_13_Aug_19_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0_RRmodel_13_Aug_19_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0.2_RRmodel_23_Jun_19' 'Neff_500_nepi_500_herit_0.5_epilvl_0.4_RRmodel_23_Jun_19' 'Neff_500_nepi_500_herit_0.5_epilvl_0.6_RRmodel_23_Jun_19' 'Neff_500_nepi_500_herit_0.5_epilvl_0.8_RRmodel_23_Jun_19' 'Neff_500_nepi_500_herit_0.5_epilvl_0_RRmodel_23_Jun_19')
RR_sims=('Neff_500_nepi_500_herit_0.5_epilvl_0.2_RRmodel_23_Jun_19' 'Neff_500_nepi_500_herit_0.5_epilvl_0.4_RRmodel_23_Jun_19' 'Neff_500_nepi_500_herit_0.5_epilvl_0.6_RRmodel_23_Jun_19' 'Neff_500_nepi_500_herit_0.5_epilvl_0.8_RRmodel_23_Jun_19' 'Neff_500_nepi_500_herit_0.5_epilvl_0_RRmodel_23_Jun_19' 'Neff_500_nepi_500_herit_0.8_epilvl_0.2_RRmodel_23_Jun_19' 'Neff_500_nepi_500_herit_0.8_epilvl_0.4_RRmodel_23_Jun_19' 'Neff_500_nepi_500_herit_0.8_epilvl_0.6_RRmodel_23_Jun_19' 'Neff_500_nepi_500_herit_0.8_epilvl_0.8_RRmodel_23_Jun_19' 'Neff_500_nepi_500_herit_0.8_epilvl_0_RRmodel_23_Jun_19' 'Neff_500_nepi_500_herit_0.2_epilvl_0.2_RRmodel_13_Aug_19_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.4_RRmodel_13_Aug_19_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.6_RRmodel_13_Aug_19_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.8_RRmodel_13_Aug_19_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0_RRmodel_13_Aug_19_final')
m16_sims=('Neff_500_nepi_500_herit_0.2_epilvl_0.2_m16model_13_Aug_19_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.4_m16model_13_Aug_19_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.6_m16model_13_Aug_19_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.8_m16model_13_Aug_19_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0_m16model_13_Aug_19_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0.2_m16model_8_Aug_19' 'Neff_500_nepi_500_herit_0.5_epilvl_0.4_m16model_8_Aug_19' 'Neff_500_nepi_500_herit_0.5_epilvl_0.6_m16model_8_Aug_19' 'Neff_500_nepi_500_herit_0.5_epilvl_0.8_m16model_8_Aug_19' 'Neff_500_nepi_500_herit_0.5_epilvl_0_m16model_8_Aug_19' 'Neff_500_nepi_500_herit_0.8_epilvl_0.2_m16model_8_Aug_19' 'Neff_500_nepi_500_herit_0.8_epilvl_0.4_m16model_8_Aug_19' 'Neff_500_nepi_500_herit_0.8_epilvl_0.6_m16model_8_Aug_19' 'Neff_500_nepi_500_herit_0.8_epilvl_0.8_m16model_8_Aug_19' 'Neff_500_nepi_500_herit_0.8_epilvl_0_m16model_8_Aug_19')
m16_sims_low_herit=('Neff_500_nepi_500_herit_0.2_epilvl_0.2_m16model_13_Aug_19_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.4_m16model_13_Aug_19_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.6_m16model_13_Aug_19_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.8_m16model_13_Aug_19_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0_m16model_13_Aug_19_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0.2_m16model_8_Aug_19' 'Neff_500_nepi_500_herit_0.5_epilvl_0.4_m16model_8_Aug_19' 'Neff_500_nepi_500_herit_0.5_epilvl_0.6_m16model_8_Aug_19' 'Neff_500_nepi_500_herit_0.5_epilvl_0.8_m16model_8_Aug_19' 'Neff_500_nepi_500_herit_0.5_epilvl_0_m16model_8_Aug_19') 
m16_sims_high_herit=('Neff_500_nepi_500_herit_0.8_epilvl_0.2_m16model_8_Aug_19' 'Neff_500_nepi_500_herit_0.8_epilvl_0.4_m16model_8_Aug_19' 'Neff_500_nepi_500_herit_0.8_epilvl_0.6_m16model_8_Aug_19' 'Neff_500_nepi_500_herit_0.8_epilvl_0.8_m16model_8_Aug_19' 'Neff_500_nepi_500_herit_0.8_epilvl_0_m16model_8_Aug_19')
fixed_sims=('Neff_500_nepi_500_herit_0.5_epilvl_0.2_fixedallepi_28_may_19' 'Neff_500_nepi_500_herit_0.5_epilvl_0.4_fixedallepi_28_may_19' 'Neff_500_nepi_500_herit_0.5_epilvl_0.6_fixedallepi_28_may_19' 'Neff_500_nepi_500_herit_0.5_epilvl_0.8_fixedallepi_28_may_19' 'Neff_500_nepi_500_herit_0.5_epilvl_0_fixedallepi_28_may_19' 'Neff_500_nepi_500_herit_0.5_epilvl_1_fixedallepi_28_may_19' 'Neff_500_nepi_500_herit_0.8_epilvl_0.2_fixedallepi_28_may_19' 'Neff_500_nepi_500_herit_0.8_epilvl_0.4_fixedallepi_28_may_19' 'Neff_500_nepi_500_herit_0.8_epilvl_0.6_fixedallepi_28_may_19' 'Neff_500_nepi_500_herit_0.8_epilvl_0.8_fixedallepi_28_may_19' 'Neff_500_nepi_500_herit_0.8_epilvl_0_fixedallepi_28_may_19' 'Neff_500_nepi_500_herit_0.8_epilvl_1_fixedallepi_28_may_19')
re_run=('Neff_500_nepi_500_herit_0.8_epilvl_0.6_RRmodel_23_Jun_19')
smalldir=('Neff_500_nepi_500_herit_0.5_epilvl_0/'   'Neff_500_nepi_500_herit_0.5_epilvl_0.2/' 'Neff_500_nepi_500_herit_0.5_epilvl_0.4/' 'Neff_500_nepi_500_herit_0.5_epilvl_0.6/' 'Neff_500_nepi_500_herit_0.5_epilvl_0.8/' 'Neff_500_nepi_500_herit_0.5_epilvl_1/')
smalldirnew=('Neff_500_nepi_500_herit_0.8_epilvl_0/'   'Neff_500_nepi_500_herit_0.8_epilvl_0.2/' 'Neff_500_nepi_500_herit_0.8_epilvl_0.4/' 'Neff_500_nepi_500_herit_0.8_epilvl_0.6/' 'Neff_500_nepi_500_herit_0.8_epilvl_0.8/' 'Neff_500_nepi_500_herit_0.8_epilvl_1/')
simdir='/projects/sysgen/users/jgrealey/Simulations/hun_k_samples/src/finalised_sep_19/'
#embed='full_batches_tanhtest_fn5000_em2500_hdf5_04_apr_19_transposed_train_split_genotypes_standardised_full_transpose_compressed.h5_transposed_validation_split_genotypes_standardised_full_transpose_compressed.h5.npy' 
embed='full_batches_fn2500_em1250_hdf5_01_apr_19_transposed_train_split_genotypes_standardised_full_transpose_compressed.h5_transposed_validation_split_genotypes_standardised_full_transpose_compressed.h5.npy'
to_run=('Neff_500_nepi_500_herit_0.5_epilvl_0.2_m16_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0.4_m16_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0.6_m16_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0.8_m16_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0_m16_model_12_Sep_19_diffseed_final')
run_diff_seed=('Neff_500_nepi_500_herit_0.8_epilvl_0.6_T_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0.6_m16_model_12_Sep_19_diffseed_final')
#levels of epistasis
fixed_sims_low_herit_same_dist=('Neff_500_nepi_500_herit_0.2_epilvl_0.2_m16_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.2_RR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.2_T_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.2_XOR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.4_m16_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.4_RR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.4_T_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.4_XOR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.6_m16_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.6_RR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.6_T_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.6_XOR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.8_m16_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.8_RR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.8_T_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.8_XOR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0_m16_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0_RR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0_T_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0_XOR_model_12_Sep_19_diffseed_final')
fixed_sims_med_herit_same_dist=('Neff_500_nepi_500_herit_0.5_epilvl_0.2_m16_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0.2_RR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0.2_T_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0.2_XOR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0.4_m16_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0.4_RR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0.4_T_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0.4_XOR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0.6_m16_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0.6_RR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0.6_T_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0.6_XOR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0.8_m16_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0.8_RR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0.8_T_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0.8_XOR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0_m16_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0_RR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0_T_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.5_epilvl_0_XOR_model_12_Sep_19_diffseed_final')
fixed_sims_high_herit_same_dist=('Neff_500_nepi_500_herit_0.8_epilvl_0.2_m16_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.8_epilvl_0.2_RR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.8_epilvl_0.2_T_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.8_epilvl_0.2_XOR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.8_epilvl_0.4_m16_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.8_epilvl_0.4_RR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.8_epilvl_0.4_T_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.8_epilvl_0.4_XOR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.8_epilvl_0.6_m16_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.8_epilvl_0.6_RR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.8_epilvl_0.6_T_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.8_epilvl_0.6_XOR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.8_epilvl_0.8_m16_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.8_epilvl_0.8_RR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.8_epilvl_0.8_T_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.8_epilvl_0.8_XOR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.8_epilvl_0_m16_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.8_epilvl_0_RR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.8_epilvl_0_T_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.8_epilvl_0_XOR_model_12_Sep_19_diffseed_final')
fixed_sims_20_herit_same_dist=('Neff_500_nepi_500_herit_0.2_epilvl_0.2_m16_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.2_RR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.2_T_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.2_XOR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.4_m16_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.4_RR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.4_T_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.4_XOR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.6_m16_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.6_RR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.6_T_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.6_XOR_model_12_Sep_19_diffseed_final')
fixed_sims_20_herit_same_dist_GPU=('Neff_500_nepi_500_herit_0.2_epilvl_0.8_m16_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.8_RR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.8_T_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0.8_XOR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0_m16_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0_RR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0_T_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.2_epilvl_0_XOR_model_12_Sep_19_diffseed_final')

emptydir=(' ')
re_run=('Neff_500_nepi_500_herit_0.8_epilvl_0.6_m16_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.8_epilvl_0.6_RR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.8_epilvl_0.6_T_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.8_epilvl_0.6_XOR_model_12_Sep_19_diffseed_final' 'Neff_500_nepi_500_herit_0.8_epilvl_0.8_T_model_12_Sep_19_diffseed_final')
time_lim=('Neff_500_nepi_500_herit_0.2_epilvl_0.6_RR_model_12_Sep_19_diffseed_final')
#turning on or off main effects
#outstr="Neff_"$nmain"_nepi_"$nepi"_herit_"$herit"_epilvl_"$epi_lvl""
cd ~/test_embeds
pipenv shell
cd /projects/sysgen/users/jgrealey/embedding/src/src/pred_nets
#/projects/sysgen/users/jgrealey/Simulations/hun_k_samples/src/standardisation_full_genotypes_23_02_19
#savestring='_28_Dec_19_diffseed_nomu_patience_100_200_reruns_seed11_GPU_fixedpatience_final'
#trainableembeds
saves=$((0))
#teststring=sleep 50s
train_file='genotype_nopretrain_final_1_Aug_2019_58001_genotypes_train_compressed.h5' 
#'transpose_genotype_split_prednets_7_04_19_38001_genotypes_train_compressed.h5'
vali_file='genotype_nopretrain_final_1_Aug_2019_2000_genotypes_vali_compressed.h5'
#'transpose_genotype_split_prednets_7_04_19_2000_genotypes_vali_compressed.h5'
test_file='genotype_nopretrain_final_1_Aug_2019_40001_test_compressed.h5'
#'transpose_genotype_split_prednets_7_04_19_39999_test_compressed.h5'
#mkdir /scratch/jgrealey/genotypes

#cpcmd=$(sbatch --job-name=cp_tr  --parsable  --wrap="cp -n $train_file /scratch/jgrealey/genotypes/")
#cpcmd2=$(sbatch --job-name=cp_va  --parsable  --wrap="cp -n $vali_file /scratch/jgrealey/genotypes/") 'Neff_500_nepi_500_herit_0.5_epilvl_0.6_m16model_8_Aug_19' 
#cpcmd3=$(sbatch --job-name=cp_te  --parsable  --wrap="cp -n $test_file /scratch/jgrealey/genotypes/")
#hparam_file=$(ls /projects/sysgen/users/jgrealey/embedding/src/src/pred_nets/final_talos_rounds/ | grep "Neff_500_nepi_500_herit_0.2_epilvl_0.8_m16_model_12_Sep_19_diffseed_final")
#echo $hparam_file
#70 for 50k
#50 for 10k
###############################################################
patience=40
ss="10k"
#ss="50k"
if [ "$ss" == "10k" ]; then
	#10k
	train_size=5000
	vali_size=1000
	test_size=4000
	echo "10k sample size with train $train_size vali $vali_size test $test_size"
	restrict_hparams="True"
	echo "restricting hyperparams $restrict_hparams"
fi
#50k
if [ "$ss" == "50k" ]; then
	train_size=29000
	vali_size=1000
	test_size=20000
	echo "50k sample size with train $train_size vali $vali_size test $test_size"
	#restrict_hparams="False"
	restrict_hparams="True"
	echo "restricting hyperparams $restrict_hparams"

fi
########################################################################
savestring="_8_may_2020_diffseed_nomu_patience_${patience}_samplesize_${ss}_train_${train_size}_vali_${vali_size}_test_${test_size}_restricthparams_GPU_final"
echo $savestring
echo "GPU JOBS "
#if [ 1 -eq 0 ]; then
#====================================================================================
#uncomment the  line below to ensure that the GPU doesn't get submitted to
#GPU=false
#====================================================================================
if ${GPU}; then
	for i in "${emptydir[@]}"
	#for i in "${run_diff_seed[@]}"
	do
		hparam_file=$(ls /projects/sysgen/users/jgrealey/embedding/src/src/pred_nets/final_talos_rounds/ | grep "${i}")
		echo "${hparam_file}"
		#exit 1
		fullsimdir=$simdir$i
    	phendir=$(find $fullsimdir -type f -name "*phenotype_num*.csv")
		string="python -u -W ignore prediction_networks_noembeds_onehot_GPU_samplesize_and_besthparams.py $train_file $vali_file $test_file $phendir  $embed "$i""$savestring" True $train_size $vali_size $test_size $hparam_file $patience $restrict_hparams"	
		#jid1=$(sbatch --output=outputs/"$i""$savestring".out --parsable ~/scripts/job_submission/gpu_neural_nets_80gb_unrestricted.sh "$string")
		echo "this one goes to gpu partition"
		

		jid1=$(sbatch --output=outputs/sample_sizes/"$i""$savestring".out --parsable ~/scripts/job_submission/gpu_neural_nets_80gb.sh "$string")
		#echo "$string"
		echo "$i"
		#currently jobs using 38G of mem
		stringoutput="outputs/"$i""$savestring".out" 
		#echo "$stringoutput"
		jid2=$(sbatch  --output="outputs/""$i""$savestring_plot_losses.out" --dependency=afterany:$jid1 ~/scripts/src/plot_losses.sh "$stringoutput")
		jid3=$(sbatch --dependency=afterany:$jid1 --ntasks=1 --mem=2g  --job-name="report" --wrap="python talos_reporting.py ${i}${savestring}"_1.csv"")
		saves=$((saves+1))
		#sleep 1
	done
fi

echo "CPU JOBS" 
#for i in "${emptydir[@]}"
# "$smalldirnew[@]}"
for i in "${fixed_sims_low_herit_same_dist[@]}"
do
	fullsimdir=$simdir$i
	hparam_file=$(ls /projects/sysgen/users/jgrealey/embedding/src/src/pred_nets/final_talos_rounds/ | grep "${i}")
	echo "${hparam_file}"

    phendir=$(find $fullsimdir -type f -name "*phenotype_num*.csv")
	#string2="python -u prediction_networks_noembeds_onehot_GPU.py $train_file $vali_file $test_file $phendir $embed "$i""$savestring" False"	
	string2="python -u -W ignore prediction_networks_noembeds_onehot_GPU_samplesize_and_besthparams.py $train_file $vali_file $test_file $phendir  $embed "$i""$savestring" False $train_size $vali_size $test_size $hparam_file $patience $restrict_hparams"	
#this line is for submitting to the normal cluster"
	#this is for submitting to the other cluster but not the GPU
	
	jid4=$(sbatch --output=outputs/sample_sizes/"$i"$savestring.out --parsable ~/scripts/job_submission/large_mem_long_80gb.sh "$string2")
	#~/scripts/job_submission/large_mem_long_80gb.sh 
	#echo "$string2"
	echo "$i"
	stringoutput2="outputs/sample_sizes/"$i""$savestring".out" 
	#echo "$stringoutput2"
	
	jid5=$(sbatch  --output="outputs/sample_sizes/""$i""$savestring_plot_losses.out" --dependency=afterany:$jid4 ~/scripts/src/plot_losses.sh "$stringoutput2")
	jid6=$(sbatch --dependency=afterany:$jid4 --ntasks=1 --mem=2g  --job-name="report" --wrap="python talos_reporting.py ${i}${savestring}"_1.csv"")
	#sleep 1
done

