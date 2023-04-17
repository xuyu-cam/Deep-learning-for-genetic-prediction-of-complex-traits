print("importing Keras")
import keras
from keras import backend as K
import tensorflow as tf
print(keras.__file__)
print("importing Random, Matplotlib, Numpy, Seaborn, sys, Zipfile, Pandas, and Sklearn")
import math
import csv
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns#; sns.set()
import sys
import os
import pandas as pd
#from sklearn.model_selection import train_test_split, cross_val_score
import zipfile
print("importing Keras Modules")
from keras.models import Sequential, Model
from keras.layers import Dense, multiply, Embedding, Reshape, Input, Lambda, Flatten ,Dropout, Dot
from keras.layers.normalization import BatchNormalization
#from keras.optimizers import Adam
from keras.initializers import Constant
from keras.models import model_from_json
print("importing Talos - May get stuck?")
import talos as ta
from talos.model.early_stopper import early_stopper
from talos.model.normalizers import lr_normalizer
#from talos.model.layers import hidden_layers
from sklearn.metrics import r2_score
from keras import regularizers
from keras.utils.np_utils import to_categorical


plt.style.use('ggplot')

from keras.layers import Layer
class MyLayer(Layer):

	def __init__(self, output_dim, **kwargs):
		self.output_dim = output_dim
		super(MyLayer, self).__init__(**kwargs)

	def build(self, input_shape):
		assert isinstance(input_shape, list)
		# Create a trainable weight variable for this layer.
		self.kernel = self.add_weight(name='kernel',
									  shape=(input_shape[0][1], self.output_dim),
									  initializer='uniform',
									  trainable=True)
		super(MyLayer, self).build(input_shape)  # Be sure to call this at the end

	def call(self, x):
		assert isinstance(x, list)
		a, b = x
		#return [K.dot(a, self.kernel) + b, K.mean(b, axis=-1)]

	def compute_output_shape(self, input_shape):
		assert isinstance(input_shape, list)
		shape_a, shape_b = input_shape
		return [(shape_a[0], self.output_dim), shape_b[:-1]]
#from keras.layers import LeakyReLU
#class LeakyReLU(LeakyReLU):
#	 def __init__(self, **kwargs):
#		 self.__name__ = "LeakyReLU"
#		 super(LeakyReLU, self).__init__(**kwargs)
def dae_model_hl(x_train,y_train, x_val,y_val,params):
	print(len(params))
	print(nsnps)
	print("number of SNPs for network is {}".format(nsnps))
	print("building autoencoder network")
	model = Sequential()
	model.add(Dense(params['first_neuron'],  activation=params['activation'], input_shape=(int(nsnps),)))
	model.add(Dropout(params['dropout']))
	model.add(BatchNormalization())
	hidden_layers(model, params, 1)
	model.add(BatchNormalization())
	model.add(Dense(params['embedding_size'],	 activation=params['activation'], name="bottleneck"))
	hidden_layers(model, params, 1)
	model.add(BatchNormalization())
	model.add(Dense(params['first_neuron'],  activation=params['activation']))
	model.add(Dropout(params['dropout']))
	model.add(BatchNormalization())

	model.add(Dense(int(nsnps),  activation=params['last_activation']))
	model.compile(optimizer=params['optimizer'](lr=lr_normalizer(params['lr'], params['optimizer'])),
				  loss=params['loss'],
				  metrics=['accuracy'])
	print("training neural network")
	callback_early_stopping=early_stopper(params['epochs'], monitor='val_loss',mode='strict')

	model.fit_generator(generator=train_generator(x_train,params['path'],params['noise'], params['batch_size']),
			validation_data=train_generator(x_train,params['path'],params['noise'], params['batch_size']),
			callbacks=[callback_early_stopping],
			samples_per_epoch=10000, nb_epoch=2,steps_per_epoch=num_train/int(params['batch_size']),validation_steps=num_val/int(params['batch_size']))
	print(model.summary())
	return out, model

def npz_headers(npz):
	"""Takes a path to an .npz file, which is a Zip archive of .npy files.
	Generates a sequence of (name, shape, np.dtype).
	"""
	with zipfile.ZipFile(npz) as archive:
		for name in archive.namelist():
			if not name.endswith('.npy'):
				continue

			npy = archive.open(name)
			version = np.lib.format.read_magic(npy)
			shape, fortran, dtype = np.lib.format._read_array_header(npy, version)
			yield name[:-4], shape, dtype

def get_input_hdf(path_input,flag,samples_to_load):
	#x=pd.read_csv(path_input,index_col=0,usecols=samples_to_load,engine='c',dtype=np.float32)#loading genotype file
	#print("last sample to read for ",flag,samples_to_load[-1])
	#older version here
	#x=pd.read_hdf(path_input,str(flag),start=samples_to_load[0],stop=samples_to_load[-1])
	x=pd.read_hdf(path_input,str(flag),start=samples_to_load[0],stop=samples_to_load[-1]+1)

	#x=pd.read_hdf(path_input,str(flag),start=samples_to_load[0],stop=samples_to_load[-1])
	
	#print(x.shape)
	#print(x.index.values)
	return (x.values,x.index)
def batch(iterable, n=1):
	while True:
		l = len(iterable)
		for ndx in range(0, l, n):
			#print(iterable[ndx:min(ndx + n, l)])
			yield iterable[ndx:min(ndx + n, l)]
'''

def batch(iterable, n=1):
	while True:
		#print("train")
		l = len(iterable)
		for ndx in range(0, l, n):
			#print(iterable[ndx:min(ndx + n+1, l+1)])
			yield iterable[ndx:min(ndx + n+1, l+1)]
'''
def batch_vali(iterable, n=1):
	while True:
		#print("vali")
		l = len(iterable)
		for ndx in range(0, l, n):
			#print(iterable[ndx:min(ndx + n+1, l+1)])
			yield iterable[ndx:min(ndx + n+1, l+1)]

def batch_test(iterable, n=1):
	while True:
		print("test")
		l = len(iterable)
		for ndx in range(0, l, n):
			#print(iterable[ndx:min(ndx + n+1, l+1)])
			yield iterable[ndx:min(ndx + n+1, l+1)]
def batch_old(iterable, n=1):
	while True:
		l = len(iterable)
		for ndx in range(0, l, n):
			#print(iterable[ndx:min(ndx + n, l)])
			yield iterable[ndx:min(ndx + n, l)]

def train_generator(input_path,batch_size,use_aux):
	#generates inputs given indexes to load, in a batch size of 64 with a given input path
	#big_batch=np.arange(num_train)
	big_batch=np.arange(train_size)
	start=1
	snp_start=1
	batches=batch(big_batch,batch_size)
	#print(num_train)
	runs=float(num_train)/batch_size
	#print(runs)
	#print(runs)
	global train_ids
	train_ids=[]
	#global train_ids
	num_runs=math.ceil(runs)
	#embeds= np.load(params['embeds'])
	#print(num_runs)
	#print(batch_size)
	#print(num_train)
	snp_runs=float(embeds.shape[0])/batch_size
	num_snp_runs=math.ceil(runs)#float(embeds.shape[0])
	snps=np.arange(embeds.shape[0])
	big_snp_batch=batch(snps,batch_size-1)#no header to laod
	#print(num_runs)
	#embeds
	while True:

		# Select files (paths/indices) for the batch
		#batch_paths = big_batch[start:start+batch_size]#np.random.choice(a = indexes, #size = batch_size,replace=False)
		#print('epoch load {} out of {}'.format(start,num_runs))
		#print(start)

		#print(num_runs)
		if start >num_runs:
			print("restarting training generator")
			batches.send('restart')
			start=0
			print("restarting batches")
		start=start+1
		'''
		if(snp_start > num_snp_runs):
			print("restarting generator")
			big_snp_batch.send('restart')
			snp_start=0
		snp_start+=1
		'''
		batch_input = []
		# Read in each input, perform preprocessing and get labels
		#print("reading training batches")
		ingoes=next(batches)
		#print(input_path)
		#print(ingoes)
		#print(big_batch)
		#print(num_runs)
		#print(next(batches))
		#sys.exit(0)
		batch_input=get_input_hdf(input_path,'train',ingoes)#next(batches))#loads the samples required with given path and array of indexes
		##previously wasn't loading the last line correctly in read_hdf
		#print(batch_input[1])
		#if start==3:
		#	sys.exit(0)
		batch_x = batch_input[0]#np.array( batch_input[0] )
		#print("train shape: ",batch_x.shape)
		pheno_input=pheno.loc[batch_input[1]]
		#print("train shape: ",pheno_input.shape)
		curr_train_ids=[]
		curr_train_ids.append(batch_input[1])
		train_ids=np.append(train_ids,curr_train_ids)
		train_ids=np.unique(train_ids)
		#print("training",train_ids[0],train_ids[-1],train_ids.shape)
		#print(train_ids)
		####
		# no longer one hot encoding, uncomment below to one-hot-encode
		###
		#batch_oh=to_categorical(batch_x, num_classes=3).reshape(batch_x.shape[0],-1)
		
		if use_aux == True:
			#yield( [batch_x.reshape(-1,nsnps), batch_x.reshape(-1,nsnps)], pheno_input.values )
			print("this method doesnt work yet, \nexiting")
			sys.exit(0)
			yield([embeds[next(big_snp_batch)],batch_x.reshape(-1,nsnps)], pheno_input.values )
			#yield([embeds ,batch_x.reshape(-1,nsnps)], pheno_input.values )
		else:
			###
			# uncomment batch_oh to line to use one hot
			###
			####yield( batch_oh, pheno_input.values )
			yield( batch_x.reshape(-1,nsnps), pheno_input.values )
		#yield( [batch_x.reshape(-1,nsnps), batch_x.reshape(-1,nsnps)], pheno_input.values )
		#yield( batch_x.reshape(-1,nsnps),pheno_input.values )
def vali_generator(input_path,batch_size,use_aux):
	#generates inputs given indexes to load, in a batch size of 64 with a given input path
	#big_batch=np.arange(num_vali)
	big_batch=np.arange(vali_size)
	start=0
	#batches=batch_vali(big_batch,batch_size)
	batches=batch(big_batch,batch_size)
	runs=float(num_vali)/batch_size
	#print(runs)
	num_runs=math.ceil(runs)
	#print(num_runs)
	#print(input_path)
	global vali_ids
	vali_ids=[]

	snp_start=0
	snp_runs=float(embeds.shape[0])/batch_size
	num_snp_runs=math.ceil(runs)#float(embeds.shape[0])
	snps=np.arange(embeds.shape[0])
	big_snp_batch=batch(snps,batch_size-1)#no header to laod
	while True:

		# Select files (paths/indices) for the batch
		#batch_paths = big_batch[start:start+batch_size]#np.random.choice(a = indexes, #size = batch_size,replace=False)
		#print("should load {} smaples".format(batch_size))
		#print("loading {} samples".format(len(batch_paths)))
		batch_input = []
		#batch_output = [] 
		# Read in each input, perform preprocessing and get labels
		
		if start >num_runs:
			print("restarting validation generator")
			batches.send('restart')
			start=0
			print("restarting batches")
		start=start+1
		'''
		if(snp_start > num_snp_runs):
			print("restarting generator")
			big_snp_batch.send('restart')
			snp_start=0
		snp_start+=1
		'''
		#print("reading batches")
		batch_input=get_input_hdf(input_path,'vali',next(batches))#loads the samples required with given path and array of indexes
		#print(batch_input.shape)
		#batch_output=get_output(output_path,batch_paths)
		# Return a tuple of (input,output) to feed the network
		batch_x = np.array( batch_input[0] )
		####
		# uncomment below to use one hot
		####
		#batch_oh=to_categorical(batch_x, num_classes=3).reshape(batch_x.shape[0],-1)
		
		#print(batch_input[1])
		#print(pheno.index)
		#print("vali")
		pheno_input=pheno.loc[batch_input[1]]
		#print(pheno_input.index)
		curr_vali_ids=[]
		curr_vali_ids.append(batch_input[1])
		vali_ids=np.append(vali_ids,curr_vali_ids)
		vali_ids=np.unique(vali_ids)
		#print("validation",vali_ids[0],vali_ids[-1],vali_ids.shape)
		#print(vali_ids)
		#print("vali shape: ",pheno_input.shape)

		#yield( batch_x.reshape(-1,nsnps), batch_x.reshape(-1,nsnps),pheno_input.values )
		#print(batch_x.reshape(batch_size,nsnps).shape)
		#print(embeds.shape)
		#print(pheno_input.shape)
		#print("yield")i
		#batch_x=to_categorical(batch_x, num_classes=3)

		#print(use_aux)
		if use_aux == True:
			#yield( [batch_x.reshape(-1,nsnps), batch_x.reshape(-1,nsnps)], pheno_input.values )
			#yield([embeds.reshape(-1,nsnps,embeds.shape[1]),batch_x.reshape(-1,nsnps)], pheno_input.values )
			yield([embeds[next(big_snp_batch)],batch_x.reshape(-1,nsnps)], pheno_input.values )
			#yield([embeds.reshape(batch_size,nsnps,embeds.shape[1]),batch_x.reshape(batch_size,nsnps)], pheno_input.values )
		else:
			# uncomment to use one hot
			#yield( batch_oh, pheno_input.values )
			yield( batch_x.reshape(-1,nsnps), pheno_input.values )


#global test_index
#test_index=np.empty(dtype='object',shape=1)#[]
def test_generator(input_path,batch_size,use_aux):
	global test_index
	#generates inputs given indexes to load, in a batch size of 64 with a given input path
	#big_batch=np.arange(num_test)
	#load only what is input
	big_batch=np.arange(test_size)
	start=0
	#batches=batch_test(big_batch,batch_size)
	batches=batch(big_batch,batch_size)
	runs=float(num_test)/batch_size
	#print(runs)
	num_runs=math.ceil(runs)
	#print(num_runs)
	#train_ids=[]
	#global train_ids
	snp_start=0
	global test_ids
	test_ids=[]
	
	#global test_index_empty
	test_index_empty=np.empty(dtype='object',shape=1)#[] 
	snp_runs=float(embeds.shape[0])/batch_size
	num_snp_runs=math.ceil(runs)#float(embeds.shape[0])
	
	snps=np.arange(embeds.shape[0])
	big_snp_batch=batch(snps,batch_size-1)
  
	while True:

		# Select files (paths/indices) for the batch
		#batch_paths = big_batch[start:start+batch_size]#np.random.choice(a = indexes, #size = batch_size,replace=False)
		#print("should load {} smaples".format(batch_size))
		#print("loading {} samples".format(len(batch_paths)))
		batch_input = []
		#batch_output = []
		# Read in each input, perform preprocessing and get labels
		'''
		if start >num_runs:
			print("restarting generator")
			batches.send('restart')
			start=0
			print("restarting batches")
		
		if(snp_start > num_snp_runs):
			print("restarting generator")
			big_snp_batch.send('restart')
			snp_start=0
		snp_start+=1
		'''
		print("reading batches")
		batch_input=get_input_hdf(input_path,'test',next(batches))#loads the samples required with given path and array of indexes
		#print(batch_input.shape)
		#batch_output=get_output(output_path,batch_paths)
		# Return a tuple of (input,output) to feed the network
		
		batch_x = np.array( batch_input[0] )
		pheno_input=pheno.loc[batch_input[1]]
		#test_index=np.concatenate(test_index,pheno_input.index.values,axis=None)#print(batch_x.reshape(batch_size,nsnps).shape)
		#global test_index
		global test_index
		test_index=np.append(test_index_empty,pheno_input.index.values,axis=None)
		print("test index",test_index.shape)
		#print(test_index)
		curr_test_ids=[]
		curr_test_ids.append(batch_input[1])
		test_ids=np.append(test_ids,curr_test_ids)
		test_ids=np.unique(test_ids)
		#print("test",test_ids[0],test_ids[-1],test_ids.shape)
		#print("test shape: ",pheno_input.shape)

		#sys.exit(0)
		#print(embeds.reshape(batch_size,nsnps,embeds.shape[1]).shape)
		#print(pheno_input.shape)
		#batch_x=to_categorical(batch_x, num_classes=3)
		#### UNCOMMENT TO ONE HOT ENCODE###
		###batch_oh=to_categorical(batch_x, num_classes=3).reshape(batch_x.shape[0],-1)

		if use_aux == True:
			#yield( [batch_x.reshape(-1,nsnps), batch_x.reshape(-1,nsnps)], pheno_input.values )
			#yield( [embeds.reshape(-1,nsnps,embeds.shape[1]), batch_x.reshape(-1,nsnps)], pheno_input.values )
			#yield([embeds.reshape(batch_size,nsnps,embeds.shape[1]),batch_x.reshape(batch_size,nsnps)], pheno_input.values )
			yield([embeds[next(big_snp_batch)],batch_x.reshape(-1,nsnps)], pheno_input.values )

		else:
			#### UNCOMMENT TO ONE HOT ENCODE yield( batch_oh, pheno_input.values )
			yield( batch_x.reshape(-1,nsnps), pheno_input.values )
			#yield( batch_x.reshape(-1,nsnps), batch_x.reshape(-1,nsnps),pheno_input.values )

def dot_help(matrices):
	a,b=matrices
	#a=K.transpose(a)
	print(K.int_shape(a),K.int_shape(b))
	dot=K.dot(a,b)
	#dot_t=K.dot(b,a)
	print(K.int_shape(dot))
	print("squeezing")
	print(K.int_shape(K.squeeze(dot,1)))
	print("performing dot product")
	return K.squeeze(dot,1)#K.dot(a,b)

def dot_product(matrices):#x, kernel):
	
	
	"""
	Wrapper for dot product operation, in order to be compatible with both
	Theano and Tensorflow
	Args:
		x (): input
		kernel (): weights

	Returns:

	"""
	x,kernel=matrices
	if K.backend() == 'tensorflow':
		# todo: check that this is correct
		return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
	else:
		return K.dot(x, kernel)
def supervised_network(x_train,y_train, x_val,y_val,params):
	print(params)
	#sys.exit(0) 
	if params['use_aux']==True:
		print("building aux net")	
		if params['if_embeds']==True:
			print("loading embeddings")
			first_input = Input(shape=(embeds.shape[1],))#input layer
			first_dense=Dense(params['pred_nodes'])(first_input)#(repest)#(first_input)#,output_shape=(nsnps,params['pred_nodes']))    
			first_dense_repeat=keras.layers.RepeatVector(nsnps)(first_dense)
			
			model_encodings=Model(inputs=first_input,outputs=first_dense_repeat)
		else:
			print("no embeddings")
			#just random and dense next layer
			first_input = Input(shape=(nsnps, ))#input layer
			first_dense = Dense(2500)(first_input)
		#load weights from weight path
		#MLP for parameters for next layer
		
		#aux_mdl=Model(first_input,first_output)
		second_input=Input(shape=(nsnps,))#same input
		second_hidden=Lambda(dot_help)([second_input,model_encodings.output])
		second_hidden=Dense(50,activation=params['activation'])(second_hidden)
		second_hidden=Dense(50,activation=params['activation'])(second_hidden)
		second_hidden=Dense(50,activation=params['activation'])(second_hidden)
		second_hidden=Dense(50,activation=params['activation'])(second_hidden)
		second_hidden=Dense(50,activation=params['activation'])(second_hidden)
	else:#no aux net
		print("no aux net")
		first_input=Input(shape=(nsnps, ))
		
		#weights = np.load(params['embeds']) 
		#first_dense = Dense(embeds.shape[1],name='embeds',weights=embeds,bias=False)(first_input)
		if params['input_reg']==True:
			first_dense=Dense(embeds.shape[1],name='embeds',activity_regularizer=regularizers.l1(float(params['lambda_reg'])),kernel_regularizer=regularizers.l1(float(params['lambda_reg'])))(first_input)
			print("with regularisation")
			#sys.exit(0)
		else:
			first_dense = Dense(embeds.shape[1],name='embeds')(first_input)
		second_hidden = Dense(params['pred_nodes'])(first_dense)
		#second_hidden=hidden_layers(model, params, 1)
		for i in range(params['hidden_layers']):
			if params['if_reg'] == True:
				if params['reg_type'] == 'l1':
					print("regularization l1")
					second_hidden = Dense(params['layer_neurons'],activation=params['activation'],activity_regularizer=regularizers.l1(float(params['lambda_reg'])),kernel_regularizer=regularizers.l1(float(params['lambda_reg'])))(second_hidden)
				else:
					print("regularization l2")
					second_hidden = Dense(params['layer_neurons'],activation=params['activation'],activity_regularizer=regularizers.l2(float(params['lambda_reg'])),kernel_regularizer=regularizers.l2(float(params['lambda_reg'])))(second_hidden)
			else:
				print("no regularisation")
				second_hidden = Dense(params['layer_neurons'],activation=params['activation'])(second_hidden)
			second_hidden=Dropout(params['dropout'])(second_hidden)		
	#reshape_final= Lambda(lambda x: K.squeeze(x,0))(second_hidden)#, -1))(second_hidden)#drops None dimension from dot product
	#reshape_final=Flatten()(second_hidden)
	#K.squeeze(second_hidden,0)#Reshape((-1,1))(second_hidden) 
	prediction=Dense(1,activation=params['last_activation'])(second_hidden)#(reshape_final)#(second_hidden)
	#print(prediction) 
	if params['use_aux']==True:
		model = Model(inputs=[first_input, second_input], outputs=prediction)
	else:#no uxilliary network
		model = Model(inputs=first_input, outputs=prediction)
		if params['trainable_embeds']== False:
			#if embeds are not trainable then set them this way
			#else keep them random intially and trainable
			model.get_layer('embeds').set_weights([embeds,np.random.random(embeds.shape[1])])#biases and weights
			print(embeds)
			#print(model.get_layer('embeds').get_weights().shape)
			print(model.get_layer('embeds').get_weights())
			#print(model.summary())
			#if params['trainable_embeds']==False:
			model.get_layer('embeds').trainable=False
			print("embeds layer trainablity:",model.get_layer('embeds').trainable)
	#print(model.layers[1])
	#model.layers[1].trainable=False
	#print(num_vali/int(params['batch_size']))
	print("validation steps")
	



	#from keras.utils.vis_utils import plot_model#keras.utils import plot_model
	#print(os.environ['PATH'].split(os.pathsep))
	#os.environ['PATH'] = os.environ['PATH'] + os.pathsep + '/home/jgrealey/.local/share/virtualenvs/embedding-3JSJ7MRQ/lib/python3.6/site-packages/graphviz/__pycache__/'#'C:\\Users\\User_Name\\AppData\\Local\\Continuum\\anaconda3\\Library\\bin\\graphviz'
	#os.environ['PROGRAMFILES'] = os.environ['PROGRAMFILES'] + os.pathsep + '/home/jgrealey/.local/share/virtualenvs/embedding-3JSJ7MRQ/lib/python3.6/site-packages/graphviz/'
	#print(os.environ['PATH'].split(os.pathsep))
	#plot_model(model, to_file=plots_dir+'model.png',show_shapes=True)
	print(model.summary())
	for layer in model.layers:
		print(layer.name)
	'''
	test_batch=np.arange(2)
	test_input_genotype=get_input_hdf(params['path'],'train',test_batch)
	#print(test_input_genotype[0])
	
	test_pheno=pheno.loc[test_input_genotype[1]]
	print(test_pheno.values.reshape(-1,1))
	test_geno=np.array(test_input_genotype[0])
	test_geno=test_geno.reshape(-1,nsnps)
	print(test_geno)
	print(embeds)
	'''
	#ada_grad = Adagrad(lr=0.1, epsilon=1e-08, decay=0.0)
	'''
	print("setting embeds")
	print(weights)
	#print(model.get_layer('embeds').get_weights().shape)
	print(model.get_layer('embeds').get_weights())
	#first_dense.set_weights(weights)
	#second_dense.set_weights(first_output_reshaped)
	#model.get_layer('embeds').set_weights(weights)
	print("should be updated")
	print(model.get_layer('embeds').get_weights())
	'''
	#data=[[embeds,test_geno],test_pheno.values]
	'''
	layer_name = 'repeat_vector_1'
	intermediate_layer_model = Model(inputs=model.input,
								 outputs=model.get_layer(layer_name).output)
	for layer in intermediate_layer_model.layers:
		print(layer.name)
	intermediate_output = intermediate_layer_model.predict([embeds,test_geno],test_pheno.values.reshape(-1,1))
	print(intermediate_output)
	sys.exit(0)
	'''
	#sys.exit(0)
	model.compile(optimizer=params['optimizer'](lr=lr_normalizer(params['lr'], params['optimizer'])), loss=params['loss'])#,metrics=['accuracy'])
	#callback_early_stopping=early_stopper(params['epochs'], monitor='val_loss',mode='strict')#moderate') 
	callback_early_stopping=early_stopper(params['epochs'],mode=[0.0,params['patience']], monitor='val_loss')#,mode='custom')#moderate')early_stopper(params['epochs'], mode='strict')
	out=model.fit_generator(train_generator(params['path'], params['batch_size'],params['use_aux']),
		verbose=2,validation_data=vali_generator(params['vali_path'], params['batch_size'],params['use_aux']),
		callbacks=[callback_early_stopping],use_multiprocessing=False,workers=0,#must be zero because read_hdf is not thread safe
		samples_per_epoch=10,#int(params['batch_size']),
		 nb_epoch=params['epochs'],steps_per_epoch=num_train/int(params['batch_size']),
		validation_steps=num_vali/int(params['batch_size']))#num_vali/int(params['batch_size']))
	#if params[''] != 'pretrained':
	#	 pass
	#samples_per_epoch is the number of batches gets sent through before an update/epoch
	'''steps_per_epoch: Total number of steps (batches of samples) to yield from generator before declaring one epoch finished and starting the next epoch. It should typically be equal to the number of unique samples of your dataset divided by the batch size.

	batch_size determines the number of samples in each mini batch. Its maximum is the number of all samples, which makes gradient descent accurate, the loss will decrease towards the minimum if the learning rate is small enough, but iterations are slower. Its minimum is 1, resulting in stochastic gradient descent: Fast but the direction of the gradient step is based only on one example, the loss may jump around. batch_size allows to adjust between the two extremes: accurate gradient direction and fast iteration. Also, the maximum value for batch_size may be limited if your model + data set does not fit into the available (GPU) memory.
	steps_per_epoch the number of batch iterations before a training epoch is considered finished. If you have a training set of fixed size you can ignore it but it may be useful if you have a huge data set or if you are generating random data augmentations on the fly, i.e. if your training set has a (generated) infinite size. If you have the time to go through your whole training data set I recommend to skip this parameter.

	validation_steps similar to steps_per_epoch but on the validation data set instead on the training data. If you have the time to go through your whole validation data set I recommend to skip this parameter.
	
	samples_per_epoch = number of times (steps_per_epoch * batch_size) gets seen before update
	if samples_per_epoch = 1, dataset is sent once through training each time

	'''
	#print(model.layers[-1].evaluate(
	#get_3rd_layer_output = K.function([model.layers[0].input],
	#							   [model.layers[1].output])
	#layer_output = get_3rd_layer_output([x])[0]
	#print(layer_output)
	return out , model
	#pass
	#build aux net
	#build pred net
	#Return correct stuff
#def build_aux_net(x_train,y_train, x_val,y_val,params):
	#takes SNPs x Samples
	#transpose x_train
	#predicts w_e for Sample i
	#if use_aux:99
	#		 embed = embedding(xt, embedding_size, dropout_rate=dropout_rate,
	#						   is_training=is_training, scope='auxembed')
	#		 We = auxnet(embed, hidden_size, dropout_rate=dropout_rate,
	#					 is_training=is_training, scope='aux_We')
	#

def main():
	#print("here")
	#a=np.array(100000*1000)
	#print("passed")
	print("main function")
	print(sys.argv)
	numcpu=4	
	snp_file=str(sys.argv[1])
	vali_file=str(sys.argv[2])
	test_file=str(sys.argv[3])
	pheno_name=str(sys.argv[4])
	embed_name=str(sys.argv[5])
	model_name=str(sys.argv[6])#"dae_test_hiddenlay_model"
	if_gpu=str(sys.argv[7])
	#global sample size variables
	global train_size,vali_size,test_size
	train_size=int(sys.argv[8])
	vali_size=int(sys.argv[9])
	test_size=int(sys.argv[10])
	
	#loaading best params from 100k
	param_to_load=str(sys.argv[11])
	#patience of single NN
	nn_patience=int(sys.argv[12])
	# if we are restricting the hpararms like lr, number of hidden layer nodes, and number of hidden layers
	restrict_hparams=str(sys.argv[13])

	print("loading best parameters from {}".format(param_to_load))
	loaded_param=pd.read_csv("/sysgen/workspace/users/jgrealey/jgrealey/embedding/src/src/pred_nets/final_talos_rounds/"+param_to_load)
	best_params=loaded_param.sort_values('val_loss',axis=0).head(n=1)
	print(best_params)
	with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
		print(best_params)#final=x.sort_values('val_loss',axis=0).head(n=1)

	for i in best_params.columns.values:
		print(i,best_params[i].values.tolist())	
	#sys.exit(0)
	print(if_gpu)
	if if_gpu =="True":
		print("\n\n\n\n\nGPU in use and {} CPU cores\n\n\n\n\n".format(numcpu))
		
		gpu_options = K.tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.1)
		#K.set_session(K.tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))
		config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': numcpu},gpu_options=gpu_options,
		intra_op_parallelism_threads=numcpu,inter_op_parallelism_threads=numcpu )
		#print("ayy lmao")
		#gpu_options = K.tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.5)
		sess = tf.Session(config=config)
		keras.backend.set_session(sess)

		#K.set_session(K.tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))
		#sys.exit(0)
	else:
		print("\n\n\n\n\nNo GPU, CPU only {} cores\n\n\n\n\n".format(numcpu))
		os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
		os.environ["CUDA_VISIBLE_DEVICES"] = ""
		config = tf.ConfigProto( device_count = {'GPU': 0 , 'CPU': numcpu},
		intra_op_parallelism_threads=numcpu,inter_op_parallelism_threads=numcpu )
		sess = tf.Session(config=config)
		keras.backend.set_session(sess)


	#rom keras import backend as K
	#K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=10, inter_op_parallelism_threads=10)))
	#print("using 32 CPUs or 64 Cores on cluster")
	if len(sys.argv)!=14:
		print("incorrect number of arguments - exiting program")
		print("1.Genotypes, train, vali, test\n2.Phenotype File\n3.Embedding File\n4.Model Name")
		sys.exit(0)
	s = random.getstate()
	print("filename for SNPs {}".format(snp_file))
	#path_input="/home/jgrealey/Simulations/ten_k_samples/test/"
	
	'''if os.path.isfile('/scratch/jgrealey/genotypes/'+snp_file) and os.path.isfile('/scratch/jgrealey/genotypes/'+vali_file):
		path_input="/scratch/jgrealey/genotypes/"
		print("loading from scratch")
	elif os.path.isfile('/scratch/jgrealey/genotypes/'+snp_file) and not os.path.isfile('/scratch/jgrealey/genotypes/'+vali_file):
		print("copy vali file")
		import subprocess
		subprocess.call(['cp','-r','/projects/sysgen/users/jgrealey/Simulations/hun_k_samples/src/standardisation_full_genotypes_23_02_19/'+str(vali_file),'/scratch/jgrealey/genotypes/'], shell = False)
	elif not os.path.isfile('/scratch/jgrealey/genotypes/'+snp_file) and os.path.isfile('/scratch/jgrealey/genotypes/'+vali_file): 
		print("copying geno file only")
		import subprocess

		subprocess.call(['cp','-r','/projects/sysgen/users/jgrealey/Simulations/hun_k_samples/src/standardisation_full_genotypes_23_02_19/'+str(snp_file),'/scratch/jgrealey/genotypes/'], shell = False)
		
	else:
		print("copying both files to scratch")
		import subprocess
		subprocess.call(['cp','-r','/projects/sysgen/users/jgrealey/Simulations/hun_k_samples/src/standardisation_full_genotypes_23_02_19/'+str(snp_file),'/scratch/jgrealey/genotypes/'], shell = False)
		#subprocess.Popen("cp " + str(snp_file) + " /scratch/jgrealey")
		#path_input="/projects/sysgen/jgrealey/Simulations/hun_k_samples/src/"
		subprocess.call(['cp','-r','/projects/sysgen/users/jgrealey/Simulations/hun_k_samples/src/standardisation_full_genotypes_23_02_19/'+str(vali_file),'/scratch/jgrealey/genotypes/'], shell = False)
	
	test_path_input='/projects/sysgen/users/jgrealey/Simulations/hun_k_samples/src/standardisation_full_genotypes_23_02_19/'
	path_input="/scratch/jgrealey/genotypes/"
	print("loading from projects/sysgen")
	#Maybe i want the standardised genotype
	'''
	path_to_copy='/scratch/jgrealey/genotypes/'#'/tmp'
	'''if os.path.isfile(path_to_copy+snp_file) and os.path.isfile(path_to_copy+vali_file):
		path_input=path_to_copy#/scratch/jgrealey/genotypes/"
		print("loading from scratch")
	elif os.path.isfile(path_to_copy+snp_file) and not os.path.isfile(path_to_copy+vali_file):
		print("copy vali file")
		import subprocess
		subprocess.call(['cp','-r','/projects/sysgen/users/jgrealey/Simulations/hun_k_samples/src/standardisation_full_genotypes_23_02_19/'+str(vali_file),path_to_copy], shell = False)
	elif not os.path.isfile(path_to_copy+snp_file) and os.path.isfile(path_to_copy+vali_file):
		print("copying geno file only")
		import subprocess

		subprocess.call(['cp','-r','/projects/sysgen/users/jgrealey/Simulations/hun_k_samples/src/standardisation_full_genotypes_23_02_19/'+str(snp_file),path_to_copy], shell = False)

	else:
		print("copying both files to scratch")
		import subprocess
		subprocess.call(['cp','-r','/projects/sysgen/users/jgrealey/Simulations/hun_k_samples/src/standardisation_full_genotypes_23_02_19/'+str(snp_file),path_to_copy], shell = False)
		#subprocess.Popen("cp " + str(snp_file) + " /scratch/jgrealey")
		#path_input="/projects/sysgen/jgrealey/Simulations/hun_k_samples/src/"
		subprocess.call(['cp','-r','/projects/sysgen/users/jgrealey/Simulations/hun_k_samples/src/standardisation_full_genotypes_23_02_19/'+str(vali_file),path_to_copy], shell = False)
	'''
	path_input=path_to_copy
	
	test_path_input='/sysgen/workspace/users/jgrealey/jgrealey/Simulations/hun_k_samples/src/standardisation_full_genotypes_23_02_19/'
	path_input=test_path_input	
	#path_pheno="/home/jgrealey/Simulations/ten_k_samples/test/"
	results_dir="/sysgen/workspace/users/jgrealey/jgrealey/embedding/hun_k_samples/results/pred_nets/"
	global plots_dir
	plots_dir="/sysgen/workspace/users/jgrealey/jgrealey/embedding/hun_k_samples/plots/pred_nets/"
	embeds_dir="/sysgen/workspace/users/jgrealey/jgrealey/embedding/hun_k_samples/results/dae/embedding_matrices/"
	#pheno_name=str(sys.argv[4])
	#embed_name=str(sys.argv[5])
	#model_name=str(sys.argv[6])#"dae_test_hiddenlay_model"
	print("checking embedding file at {}{}".format(embeds_dir,embed_name))
	if os.path.exists(embeds_dir+embed_name):
		if_embeds=True
	else:
		if_embeds=False
	experiment_name=model_name+"_"+snp_file+pheno_name
	print(experiment_name)
	print(model_name)
	print(pheno_name)
	input_file=path_input+snp_file

	#train_ids=pd.read_hdf(input_file,'train',columns=[0]).index.values#start=samples_to_load[0],stop=samples_to_load[-1])
	
	#if file has been transposed samXSNP
	import h5py
	print(input_file)
	chunk_file = h5py.File(input_file, "r")
	chunk_vali_file=h5py.File(path_input+vali_file, "r")
	chunk_test_file=h5py.File(test_path_input+test_file, "r")
	chunk=chunk_file['train']
	chunk_vali=chunk_vali_file['vali']
	chunk_test=chunk_test_file['test']
	print("reading samples from {}".format(input_file))
	#this is of all available
	global num_train
	num_train=chunk['/train/axis1'].shape[0]#numsam=pd.read_csv(input_file,usecols=[0])#(input_file,nrows=1,index_col=0)
	if train_size>=num_train:
		print("training size is max")
		train_size=num_train
	print("reading SNPs from {}".format(input_file))
	#snps=pd.read_csv(input_file,nrows=1,index_col=0)#(input_file,usecols=[0])
	global nsnps
	nsnps=chunk['/train/axis0'].shape[0]
	####
	# uncomment this line for one hot encoding 
	#####nsnps=3*nsnps
	####
	global num_vali
	num_vali=chunk_vali['/vali/axis1'].shape[0]
	global num_test
	num_test=chunk_test['/test/axis1'].shape[0]
	#else sNPxSAM
	if vali_size>=num_vali:
		print("validation size is max")
		vali_size=num_vali
	if test_size>=num_test:
		print("testing size is max")
		test_size=num_test
	print(nsnps,num_train,num_vali,num_test)
	print(train_size,vali_size,test_size)
	#print(train_ids)
	all_train=num_train
	all_vali=num_vali
	all_test=num_test
	num_train=train_size
	num_vali=vali_size
	num_test=test_size
	print("loading from")
	print(input_file)
	print(path_input+vali_file)
	print(test_path_input+test_file)
	#sys.exit(0)
	test_batch=np.arange(100)
	test_input_genotype=get_input_hdf(input_file,'train',test_batch)
	global pheno
	pheno=pd.read_csv(pheno_name,index_col=0)
	print("scanning")

	print("setting up parameter dictionary")

	if 'linear' in best_params['activation'].values:
		acti=[keras.activations.linear]
	else:
		acti=[keras.activations.relu]
	if 'linear' in best_params['last_activation'].values:
		lastacti=[keras.activations.linear]
	else:
		lastacti=[keras.activations.relu]
	
	print("restricting hparams?",restrict_hparams)
	print("model patience",nn_patience)	
	if "True" in restrict_hparams:#=="True":
		#str(sys.argv[13])
		print("restricting hparams")
		p_best_loaded= {
		'lr': [0.1],#best_params['lr'].to_numpy(dtype=float).tolist(),
		'first_neuron':[300],#best_params['first_neuron'].values.tolist(),
		'if_embeds':[True],#best_params['if_embeds'].values.tolist(),
		'batch_size':[100],#best_params['batch_size'].values.tolist(),
		'epochs':[10000],#best_params['epochs'].values.tolist(),
		'noise':[0.5],#best_params['noise'].values.tolist(),
		'dropout':[0.2],#best_params['dropout'].values.tolist(),
		'trainable_embeds':[True],#best_params['trainable_embeds'].values.tolist(),
		#10k
		'hidden_layers':[1],
		'embeds':[embeds_dir+embed_name],#best_params['embeds'].values.tolist(),#]'/projects/sysgen/users/jgrealey/embedding/hun_k_samples/results/dae/embedding_matrices/full_batches_fn2500_em1250_hdf5_01_apr_19_transposed_train_split_genotypes_standardised_full_transpose_compressed.h5_transposed_validation_split_genotypes_standardised_full_transpose_compressed.h5.npy'],
		'testnet':[True],#best_params['testnet'].values.tolist(),
		'pred_nodes':[100],#best_params['pred_nodes'].values.tolist(),
		'optimizer':[keras.optimizers.Adam],#only adam here so just hardcode itest_params['optimizer'].values.tolist(),
		'patience':[int(nn_patience)],
		'if_reg':[False],#best_params['if_reg'].values.tolist(),
		'lambda_reg':[0.001],#best_params['lambda_reg'].values.tolist(),
		'reg_type':['l1'],#best_params['reg_type'].values.tolist(),
		'input_reg':['False'],#best_params['input_reg'].values.tolist(),
		'activation':acti,#taken from best one
		'use_aux':[False],#best_params['use_aux'].values.tolist(),
		#10k samples size
		'layer_neurons':[100],
		'path':[input_file],
		'vali_path':[path_input+vali_file],#'
		'loss':['mse'],#best_params['loss.1'].values.tolist(),
		'last_activation':[keras.activations.linear]}
	else:
		print("loading best unrestricted params with patience:",nn_patience)
		p_best_loaded= {
		'first_neuron':[300],#best_params['first_neuron'].values.tolist(),
		'if_embeds':[True],#best_params['if_embeds'].values.tolist(),
		'batch_size':[100],#best_params['batch_size'].values.tolist(),
		'epochs':[10000],#best_params['epochs'].values.tolist(),
		'noise':[0.5],#best_params['noise'].values.tolist(),
		'dropout':[0.2],#best_params['dropout'].values.tolist(),
		'trainable_embeds':[True],#best_params['trainable_embeds'].values.tolist(),
		'if_reg':[False],#best_params['if_reg'].values.tolist(),
		'lambda_reg':[0.001],#best_params['lambda_reg'].values.tolist(),
		'reg_type':['l1'],#best_params['reg_type'].values.tolist(),
		'input_reg':['False'],#best_params['input_reg'].values.tolist(),
		'activation':acti,#taken from best one
		'use_aux':[False],#best_params['use_aux'].values.tolist(),
		'path':[input_file],
		'vali_path':[path_input+vali_file],#'
		'loss':['mse'],#best_params['loss.1'].values.tolist(),
		'embeds':[embeds_dir+embed_name],#best_params['embeds'].values.tolist(),#]'/projects/sysgen/users/jgrealey/embedding/hun_k_samples/results/dae/embedding_matrices/full_batches_fn2500_em1250_hdf5_01_apr_19_transposed_train_split_genotypes_standardised_full_transpose_compressed.h5_transposed_validation_split_genotypes_standardised_full_transpose_compressed.h5.npy'],
		'testnet':[True],#best_params['testnet'].values.tolist(),
		'pred_nodes':[100],#best_params['pred_nodes'].values.tolist(),
		'optimizer':[keras.optimizers.Adam],#only adam here so just hardcode itest_params['optimizer'].values.tolist(),
		
		'patience':[int(nn_patience)],
		#'patience':[80],
		#10k samples size
		#'layer_neurons':[100],
		'layer_neurons':best_params['layer_neurons'].to_numpy(dtype=int).tolist(),
		'lr': best_params['lr'].to_numpy(dtype=float).tolist(),
		'hidden_layers':best_params['hidden_layers'].to_numpy(dtype=int).tolist(),

		'last_activation':[keras.activations.linear]}
	print("hparam|","best from 100k|","utilised")
	for i in ['lr','hidden_layers','layer_neurons']:
		print(i,best_params[i].values.tolist(),p_best_loaded[i])
	
	#for i in p_best_loaded:
	#	print(i,best_params[i].values.tolist(),p_best_loaded[i])
	#sys.exit(0)	
	pold= {'lr': (0.1, 1.0, 3),#,'lr':[0.7],
	'first_neuron':[300],
	'if_embeds':[True],
	'batch_size':[100],
	'epochs':[5000],
	'noise':[0.5],
	'dropout':[0.2],
	'trainable_embeds':[False],
	'hidden_layers':[1, 2],
	'embeds':[embeds_dir+embed_name],#]'/projects/sysgen/users/jgrealey/embedding/hun_k_samples/results/dae/embedding_matrices/full_batches_fn2500_em1250_hdf5_01_apr_19_transposed_train_split_genotypes_standardised_full_transpose_compressed.h5_transposed_validation_split_genotypes_standardised_full_transpose_compressed.h5.npy'],
	'testnet':[True],
	'pred_nodes':[100],
	'optimizer':[keras.optimizers.Adam],
	'patience':[4],
	'if_reg':[False],#[True,False]
	'lambda_reg':[0.01],
	'reg_type':['l1'],
	'activation':[keras.activations.elu,keras.activations.relu],
	'use_aux':[False],
	'layer_neurons':[1000,500],
	'path':[input_file],
	'vali_path':[path_input+vali_file],#'/scratch/jgrealey/genotypes/transpose_genotype_split_prednets_7_04_19_38001_genotypes_train_compressed.h5'],
	'loss':['mse'],
	'last_activation':[keras.activations.linear]}


	pnew_oct2019= {'lr': [0.1,0.4],#](0.1, 1.0, 3),#,'lr':[0.7],
		'first_neuron':[300],
	'if_embeds':[True],
	'batch_size':[100],
	'epochs':[10000],
	#'epochs':[1],
	'noise':[0.5],
	'dropout':[0.2],
	'trainable_embeds':[True],
	'hidden_layers':[1,2,3],
	'embeds':[embeds_dir+embed_name],#]'/projects/sysgen/users/jgrealey/embedding/hun_k_samples/results/dae/embedding_matrices/full_batches_fn2500_em1250_hdf5_01_apr_19_transposed_train_split_genotypes_standardised_full_transpose_compressed.h5_transposed_validation_split_genotypes_standardised_full_transpose_compressed.h5.npy'],
	'testnet':[True],
	'pred_nodes':[100],
	'optimizer':[keras.optimizers.Adam],
	#50%herit and 80% herit
	#'patience':[100,200],
	#for 20% herit
	'patience':[50,70],
	#testing patiences
	#'patience':[15,30],
	#'patience':[5,10],
	'if_reg':[False],#[True,False]
	'lambda_reg':[0.001],
	'reg_type':['l1'],
	'input_reg':[False],
	'activation':[keras.activations.linear,keras.activations.relu],
	'use_aux':[False],
	'layer_neurons':[100,1000],
	'path':[input_file],
	'vali_path':[path_input+vali_file],#'/scratch/jgrealey/genotypes/transpose_genotype_split_prednets_7_04_19_38001_genotypes_train_compressed.h5'],
	'loss':['mse'],
	'last_activation':[keras.activations.linear]}

	
	pnew_oct2019_test= {'lr': [0.1],#](0.1, 1.0, 3),#,'lr':[0.7],
		'first_neuron':[300],
	'if_embeds':[True],
	'batch_size':[100],
	'epochs':[1],
	#'epochs':[1],
	'noise':[0.5],
	'dropout':[0.2],
	'trainable_embeds':[True],
	'hidden_layers':[1],
	'embeds':[embeds_dir+embed_name],#]'/projects/sysgen/users/jgrealey/embedding/hun_k_samples/results/dae/embedding_matrices/full_batches_fn2500_em1250_hdf5_01_apr_19_transposed_train_split_genotypes_standardised_full_transpose_compressed.h5_transposed_validation_split_genotypes_standardised_full_transpose_compressed.h5.npy'],
	'testnet':[True],
	'pred_nodes':[100],
	'optimizer':[keras.optimizers.Adam],
	#50%herit and 80% herit
	#'patience':[100,200],
	#for 20% herit
	#'patience':[50,70],
	
	#testing patiences
	'patience':[2],
	'if_reg':[False],#[True,False]
	'lambda_reg':[0.001],
	'reg_type':['l1'],
	'input_reg':[False],
	'activation':[keras.activations.linear,keras.activations.relu],
	'use_aux':[False],
	'layer_neurons':[100],
	'path':[input_file],
	'vali_path':[path_input+vali_file],#'/scratch/jgrealey/genotypes/transpose_genotype_split_prednets_7_04_19_38001_genotypes_train_compressed.h5'],
	'loss':['mse'],
	'last_activation':[keras.activations.linear]}


	pnew= {'lr': (0.1, 1.0, 3),#,'lr':[0.7],
		'first_neuron':[300],
	'if_embeds':[True],
	'batch_size':[100],
	'epochs':[10000],
	#'epochs':[1],
	'noise':[0.5],
	'dropout':[0.2],
	'trainable_embeds':[True],
	'hidden_layers':[1, 2],
	'embeds':[embeds_dir+embed_name],#]'/projects/sysgen/users/jgrealey/embedding/hun_k_samples/results/dae/embedding_matrices/full_batches_fn2500_em1250_hdf5_01_apr_19_transposed_train_split_genotypes_standardised_full_transpose_compressed.h5_transposed_validation_split_genotypes_standardised_full_transpose_compressed.h5.npy'],
	'testnet':[True],
	'pred_nodes':[100],
	'optimizer':[keras.optimizers.Adam],
	'patience':[15,20],
	'if_reg':[False],#[True,False]
	'lambda_reg':[0.001],
	'reg_type':['l1'],
	'input_reg':[False],
	'activation':[keras.activations.linear,keras.activations.relu],
	'use_aux':[False],
	'layer_neurons':[700,500],
	'path':[input_file],
	'vali_path':[path_input+vali_file],#'/scratch/jgrealey/genotypes/transpose_genotype_split_prednets_7_04_19_38001_genotypes_train_compressed.h5'],
	'loss':['mse'],
	'last_activation':[keras.activations.linear]}
	ptest= {'lr': [0.1,0.001,0.01,0.00005,0.2,0.002,0.02],#,'lr':[0.7],
		'first_neuron':[300],
	'if_embeds':[True],
	'batch_size':[100],
	'epochs':[1],
	#'epochs':[1],
	'noise':[0.5],
	'dropout':[0.2],
	'trainable_embeds':[True],
	'hidden_layers':[1],
	'embeds':[embeds_dir+embed_name],#]'/projects/sysgen/users/jgrealey/embedding/hun_k_samples/results/dae/embedding_matrices/full_batches_fn2500_em1250_hdf5_01_apr_19_transposed_train_split_genotypes_standardised_full_transpose_compressed.h5_transposed_validation_split_genotypes_standardised_full_transpose_compressed.h5.npy'],
	'testnet':[True],
	'pred_nodes':[100],
	'optimizer':[keras.optimizers.Adam],
	'patience':[10],
	'if_reg':[False],#[True,False]
	'lambda_reg':[0.01],
	'reg_type':['l1'],
	'input_reg':[True],
	'activation':[keras.activations.linear],
	'use_aux':[False],
	'layer_neurons':[500],
	'path':[input_file],
	'vali_path':[path_input+vali_file],#'/scratch/jgrealey/genotypes/transpose_genotype_split_prednets_7_04_19_38001_genotypes_train_compressed.h5'],
	'loss':['mse'],
	'last_activation':[keras.activations.linear]}

	p = {'lr': [0.1],#0.1, 1.0, 3),#'lr': (0.1, 10),
	'first_neuron':[ 300],
	'if_embeds':[True],#if to use embeds
	'batch_size': [100],#small due to multi load and also two reads
	'epochs': [5000],
	'noise':[0.5],
	'dropout': [0.2],
	'trainable_embeds':[False],#if embeds are trainbable
	'hidden_layers':[1],
	'embeds':[embeds_dir+embed_name],#]'/projects/sysgen/users/jgrealey/embedding/hun_k_samples/results/dae/embedding_matrices/full_batches_fn2500_em1250_hdf5_01_apr_19_transposed_train_split_genotypes_standardised_full_transpose_compressed.h5_transposed_validation_split_genotypes_standardised_full_transpose_compressed.h5.npy'],
	'testnet':[True],
	'pred_nodes':[100],
	'optimizer':[keras.optimizers.Adam],
 
	'patience':[2],
	'if_reg':[True,False],
	'lambda_reg':[0.01],#regularisation parameter
	'reg_type':['l1'],
	'input_reg':[False],
	'if_reg':[False],#[True,False]
	'lambda_reg':[0.001],
	'reg_type':['l1'],
	'input_reg':[False],
	'activation':[keras.activations.linear,keras.activations.relu],
	'use_aux':[False],
	'layer_neurons':[100,1000],
	'path':[input_file],
	'vali_path':[path_input+vali_file],#'/scratch/jgrealey/genotypes/transpose_genotype_split_prednets_7_04_19_38001_genotypes_train_compressed.h5'],
	'loss':['mse'],
	'last_activation':[keras.activations.linear]}


	dummy_data=np.arange(10)#need to pass dummy data because we use generators
	global embeds
	embeds= np.load(embeds_dir+embed_name)
		
	num_sam=num_train + num_vali + num_test
	percent_train=float(num_train)/num_sam
	percent_vali=float(num_vali)/num_sam
	percent_test=float(num_test)/num_sam
	
	from talos.utils.gpu_utils import parallel_gpu_jobs
	# split GPU memory in two for two parallel jobs
	if if_gpu =="True":
		print("activating GPU utils from Talos")
		from talos.utils.gpu_utils import parallel_gpu_jobs
		parallel_gpu_jobs(1)#0.2
	oldseed=10
	#puse=pnew_oct2019#_test
	puse=p_best_loaded
	if puse == p_best_loaded:
		fraction_downsample=1
	else:
		fraction_downsample=0.3
	print("learning rate adjustment")
	print(p_best_loaded['lr'])
	#p_best_loaded['lr']=[float(i)*0.5 for i in p_best_loaded['lr']]#float(p_best_loaded['lr'])*0.5
	print(p_best_loaded['lr'])
	
	print("old and new dictionary")
	for key in pnew_oct2019:
		if pnew_oct2019[key]!=p_best_loaded[key]:
			print(key,pnew_oct2019[key],p_best_loaded[key])
		if type(pnew_oct2019[key])!=type(p_best_loaded[key]):
			print("hello")
		print(type(pnew_oct2019[key]),type(p_best_loaded[key]))
			#type(p_best_loaded[key])=type(pnew_oct2019[key])
			#print(type(pnew_oct2019[key]),type(p_best_loaded[key]))
			#print("should be the same above")
	print(best_params)
	#sys.exit(0)
		
	h = ta.Scan(x=dummy_data,x_val=dummy_data, y=dummy_data,y_val=dummy_data, params=puse, model=supervised_network,
	experiment_name=model_name,#'dae_test_hiddenlay_model',
	#experiment_no='1',#)
	seed=11,
	
	fraction_limit=fraction_downsample,
	print_params=True,disable_progress_bar=False)

	with open('parameter_dictionaries/parameters_dictionary'+model_name+'dict.csv', 'w') as csv_file:
		writer = csv.writer(csv_file)
		for key, value in puse.items():
			writer.writerow([key, value])

	from talos.utils.best_model import best_model, activate_model
	scan_object=h
	print(scan_object.data)
	#print(train_ids)
	#print(vali_ids)
	#print(test_index)

	#sys.exit(0)
	#from talos.utils.string_cols_to_numeric import string_cols_to_numeric
	#scan_object.data = string_cols_to_numeric(scan_object.data)
	#print(scan_object.datai)
	print(scan_object.data['val_loss'])
	print(best_model(scan_object,'val_loss',asc=True))
	model=activate_model(scan_object, best_model(scan_object, 'val_loss', asc=True))#NOTE: for loss 'asc' should be True#.predict(X)
	preds=model.predict_generator(test_generator(test_path_input+test_file, 100 ,False),steps=math.ceil(num_test/100),workers=0, use_multiprocessing=True, verbose=0)
	print("BEST model here!")
	print(preds)
	best_idx=best_model(scan_object, 'val_loss', asc=True)	
	print(best_model(scan_object, 'val_loss', asc=True))
	print("BEST model above")
	print("best model params")
	print(scan_object.data.iloc[best_idx])
	#test_pheno=pheno.loc[test_index[test_index != np.array(None)]].values
	test_pheno=pheno.loc[test_ids].values
	print(preds,test_pheno)#pheno.loc[test_index[test_index != np.array(None)]].values)
	r2=r2_score(test_pheno,preds)
	print(r2)
	print("saving at ",results_dir+model_name)
	print("model configuration")
	model_config=model.get_config()
	print(model_config)
	print("model configuration above")
	np.savetxt(results_dir+model_name+'_true_phenotype.out', test_pheno, delimiter=',')
	np.savetxt(results_dir+model_name+'_predicted_phenotype.out', preds, delimiter=',')
	#You could use best_model() to identify the id of the best model, and then model = activate_model(id) and finally model.predict(x).
	all_train_ids=np.append(train_ids,vali_ids)
	
	print(all_train_ids)
	#test_index=test_index[test_index != np.array(None)]
	#print(test_index)
	np.savetxt(results_dir+model_name+"train_and_vali_indexes",all_train_ids,fmt='%s',delimiter=',')
	np.savetxt(results_dir+model_name+"test_indexes",test_ids,delimiter=',',fmt='%s')
	print("indexes saved to "+results_dir+model_name+"indexes")
	print("train index length",train_ids.shape)
	print("validation index length",vali_ids.shape)
	print("test index length",test_ids.shape)	
	
	from scipy.stats import spearmanr
	spearman,pval=spearmanr(test_pheno,preds)
	r2=r2_score(test_pheno,preds)
	print("rsq:",r2)
	print("spearman",spearman)
	print("plotting at, ",plots_dir)
	fig,ax = plt.subplots()
	ax.scatter(test_pheno, preds,marker="+",s=0.5)
	ax.plot([test_pheno.min(), test_pheno.max()], [test_pheno.min(), test_pheno.max()], 'k--', lw=4)
	ax.annotate("Rsq: {:.2f}\nR: {:.2f}\nP: {:.2E}".format(r2,spearman,pval),xy=(test_pheno.max()-1,preds.max()-1))
	#ax.annotate("Rsq - {:.2f}".format(r2),xy=(test_pheno.max(),preds.max()))
	ax.set_xlabel('True Phenotype')
	ax.set_ylabel('Predicted Phenotype')
	fig.savefig(plots_dir+model_name+"_prediction_plots.pdf", bbox_inches='tight')
	fig.show()
	print("saving model")
	# serialize model to JSON
	model_json = model.to_json()
	print("model serialised")
	with open("/sysgen/workspace/users/jgrealey/jgrealey/embedding/hun_k_samples/models/model_"+model_name+".json", "w") as json_file:
		json_file.write(model_json)
	print("model written")
	# serialize weights to HDF5
	model.save_weights("/sysgen/workspace/users/jgrealey/jgrealey/embedding/hun_k_samples/models/model_"+model_name+".h5")
	
	print("Saved model to disk")
	print("model saved as\n","/sysgen/workspace/users/jgrealey/jgrealey/embedding/hun_k_samples/models/model_{}".format(model_name))
if __name__ == "__main__":
	main()
