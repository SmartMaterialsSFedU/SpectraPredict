import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from DeepTimeSeries.models.RNN2Dense import *
from DeepTimeSeries.models.Seq2Seq import *
from DeepTimeSeries.utils import series_to_superviesed, load_time_series_model
import numpy as np
import matplotlib.pyplot as plt

# hyperparameters =======================
project_name = 'DTSeries'               # will be used to save model
task = 'train'   					# 'train' / 'predict'
n_memory_steps = 3					# time steps for encoder
n_forcast_steps = 3		    		# time steps for decoder
train_split = 0.8		    		# proportion as train set
batch_size = 16						# batch size for training
epochs = 1							# epochs for training
test_model = 'RNN2Dense_2'    		# 'RNN2Dense' / 'Seq2Seq_1' / 'Seq2Seq_2'
cell = 'SimpleRNN'           		# 'SimpleRNN' / 'LSTM' / 'GRU'
is_augmentation = False     		# if True, x^2, x^3 will be included
# =======================================

#----------sample Dataset
#df = pd.read_csv('clean_data.csv', index_col=0, header=0)
#df.info()
#values = df.values
#------------------------

#---------Spectra Dataset-----------------
spectra = pd.read_table('PCA_ref_shift.txt',  delim_whitespace=True, header=None)
energies = spectra.loc[0]
spectra = spectra.drop([0])   #drop column 0 - energies
spectra = spectra.T
values = spectra.values

# data augmentation, add x^2
is_augmentation = True
if is_augmentation:
       values = np.hstack((values,values**2))


# scale data between 0 ~ 1 for better training results
data_scaler = MinMaxScaler(feature_range=(0, 1))
scaled_values = data_scaler.fit_transform(values)

# convert to supervised learning compatible format
x_timeseries = scaled_values
y_timeseries = scaled_values[:,1].reshape(-1,1)

x_train, y_train, x_test, y_test = \
	series_to_superviesed(x_timeseries, y_timeseries, n_memory_steps, n_forcast_steps, split = 0.8)
print('\nsize of x_train, y_train, x_test, y_test:')
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape )

# train model & inference
if task == 'train':
	# build model
	if test_model == 'RNN2Dense_1':
		model = RNN2Dense_1(x_train.shape[1:], y_train.shape[1:], cell, 300, (20,))
	elif test_model == 'RNN2Dense_2':
		model = RNN2Dense_2(x_train.shape[1:], y_train.shape[1:], cell, 300, (20,))
	elif test_model == 'Seq2Seq_1':
		model = Seq2Seq_1(x_train.shape[1:], y_train.shape[1:], cell, 300)
	elif test_model == 'Seq2Seq_2':
		model = Seq2Seq_2(x_train.shape[1:], y_train.shape[1:], cell, 300)
	print(model.summary())
	# compile model
	model.compile()
	# train model
	history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
	   verbose=2, validation_data=(x_test,y_test))
	# save model
	model.save(project_name)
	plt.plot(y_timeseries[:,0],'g',label='Train')


elif task == 'predict':
	# reload model
	model = load_time_series_model(project_name)
	# predict data
	y_pred = model.predict(x_test)
	# plot results
	for n in range(n_forcast_steps):
		plt.subplot(n_forcast_steps,1,n+1)
		plt.plot(y_test[500:800,n,:],'b', label = 'True')
		plt.plot(y_pred[500:800,n,:],'r', label = 'Predict')
		plt.legend()
	plt.show()
plt.legend()
plt.show()

