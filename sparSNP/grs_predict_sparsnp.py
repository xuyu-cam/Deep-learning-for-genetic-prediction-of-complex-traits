#import statsmodels.api as sm
import numpy as np
import sys
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
import statsmodels.api as sm
import os
from sklearn.linear_model import LinearRegression



def main():
	if len(sys.argv)!=3:
		print("incorrect number of arguments\nexiting program")
		sys.exit(0)
	plot_dir="/projects/sysgen/users/jgrealey/embedding/hun_k_samples/plots/sparsnp/"
	#/projects/sysgen/users/jgrealey/embedding/hun_k_samples/plots/grs/"
	results_dir="/projects/sysgen/users/jgrealey/embedding/hun_k_samples/results/sparsnp/"
	#"/projects/sysgen/users/jgrealey/embedding/hun_k_samples/results/grs/"
	grs_file=sys.argv[1]
	pheno_file=sys.argv[2]
	print(grs_file,pheno_file)
	#	note load in GRS and split pheno file
	grs=pd.read_csv(grs_file,sep=' ',index_col=0,usecols=[0,5])#,usecols=[4])
	pheno=pd.read_csv(pheno_file,index_col=0,sep=' ',header=None,usecols=[0,2])
	print(grs.head)
	print(pheno.head)
	head, tail= os.path.split(str(grs_file))
	top, bottom=os.path.split(str(pheno_file))
	top, bottom = os.path.split(str(top))
	print(bottom)
	print(tail)
	print(grs.values)
	
	sns.distplot(grs.values)
	plt.xlabel("GRS")
	plt.ylabel("Density")
	plt.title("GRS Distribution for Test Set")
	plt.savefig(plot_dir+tail+bottom+"GRS_sparsnp_density.pdf",bbox_inches='tight')

	modelgrs=sm.OLS(pheno,sm.add_constant(grs.values))#all_genetics_epi=
	#modelgrs=sm.OLS(pheno[1:5000],sm.add_constant(grs.values[1:5000]))
	reg = LinearRegression(fit_intercept=True).fit(grs.values, pheno)
	linr2=reg.score(grs.values, pheno)#X, y)
	#modelenv=sm.OLS(pheno,environmentaleffect)
	resultsgrs=modelgrs.fit()
	smr2=resultsgrs.rsquared
	print(resultsgrs.summary())
	print(smr2,linr2)
	pvals=resultsgrs.pvalues
	print(pvals[1])
	numsplits=50
	from sklearn.model_selection import KFold, cross_val_score
	#X = ["a", "a", "a", "b", "b", "c", "c", "c", "c", "c"]
	regr = LinearRegression()
	#k_fold = KFold(n_splits=numsplits)
	#for train_indices, test_indices in k_fold.split(grs):
	#	print('Train: %s | test: %s' % (train_indices, test_indices))
	#fitscores=[regr.fit(grs.values[train], pheno.values[train]).score(grs.values[test], pheno.values[test]) for train, test in k_fold.split(grs)]
	#predictions=[regr.fit(grs.values[train], pheno.values[train]).predict(grs.values[test]) for train, test in k_fold.split(grs)]
	#trues=[pheno.values[test] for train,test in k_fold.split(grs)]
	#print(predictions,trues)

	#fitscores=[]
	#r2s=[]
	#rhos=[]
	#predicting and getting R2 and R from predictions
	from scipy.stats import spearmanr
	from sklearn.metrics import r2_score

	with open(results_dir+"/fitting_sparsnp_GRS_"+str(tail)+"on_pheno_"+str(bottom)+".csv", 'w') as f:
		f.write(resultsgrs.summary().as_csv())	
	score=regr.fit(grs.values, pheno.values).score(grs.values, pheno.values)#fitting
	predict = regr.fit(grs.values, pheno.values).predict(grs.values) #predicting
	spearman,pval=spearmanr(pheno,predict)
	r2=r2_score(pheno,predict)
	print("rsq:",r2)
	print("spearman",spearman)

	'''
	k_fold = KFold(n_splits=numsplits)

	for train, test in k_fold.split(grs):
		#fitscores=[regr.fit(grs.values[train], pheno.values[train]).score(grs.values[test], pheno.values[test]) for train, test in k_fold.split(grs)]
		score=regr.fit(grs.values[train], pheno.values[train]).score(grs.values[test], pheno.values[test])
		predict = regr.fit(grs.values[train], pheno.values[train]).predict(grs.values[test])  
		trues=pheno.values[test]
		#print(predict,trues)
		spearman,pval=spearmanr(trues,predict)
		r2=r2_score(trues,predict)
		print("rsq:",r2)
		print("spearman",spearman)
		r2s.append(r2)
		rhos.append(spearman)
	print("\nfitting with {} splits".format(numsplits))
	print("mean Rsq\tstdev Rsq")
	print(np.mean(r2s),np.std(r2s))
	print("mean Rho\tstdev Rho")

	print(np.mean(rhos),np.std(rhos))
	'''
	#print(fitscores)
	

	fig,ax = plt.subplots()
	ax.scatter(pheno, predict,marker="+",s=0.5)
	ax.plot([pheno.values.min(), pheno.values.max()], [pheno.values.min(), pheno.values.max()], 'k--', lw=4)
	ax.annotate("Rsq: {:.2f}\nR: {:.2f}\nP: {:.2E}".format(r2,spearman,pval),xy=(pheno.max()-1,predict.max()-1))
	#ax.annotate("Rsq - {:.2f}".format(r2),xy=(test_pheno.max(),preds.max()))
	ax.set_xlabel('True Phenotype')
	ax.set_ylabel('Predicted Phenotype')
	fig.savefig(plot_dir+tail+bottom+"_prediction_sparsnp_plots.pdf", bbox_inches='tight')
	fig.show()

if __name__=="__main__":
	main()
