import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from SpectraTimeSeries.SeriesModels.RNN2Dense import *
from SpectraTimeSeries.SeriesModels.Seq2Seq import *
from SpectraTimeSeries.Utils.utils import timeseries_to_supervised,load_timeseries_model
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys

parser = argparse.ArgumentParser(description="SpectraPredict Framework")
parser.add_argument(
    "--project_name", default="SpectraPredict", type=str, help="Name of SpectraPredict Model",
)
parser.add_argument(
    "--task", default="train", type=str, help="train / predict (default:train)" )
parser.add_argument(
    "--n_memory_steps",  default=1, type=int, help="number of steps >=1 (default:1)")
parser.add_argument(
    "--n_forcast_steps", default=1, type=int, help="number of forecast steps >=1 (default:1)")
parser.add_argument(
    "--train_split", default=0.2, type=float, help=" train / split ratio from 0.1 to 0.9 (default:0.2)"
)
parser.add_argument(
    "--batch_size", default=1, type=int, help="batch size (default:1)")
parser.add_argument(
    "--epochs", default=1, type=int, help="epochs >=1 (default: 1)"
)
parser.add_argument(
    "--model", default="RNN2Dense_2", type=str, help="time series model RNN2Dense_2/Seq2Seq_1/Seq2Seq_2 (default:RNN2Dense_2)")
parser.add_argument(
    "--cell", default="SimpleRNN",  type=str,  help="type of cell: SimpleRNN/LSTM/GRU (default:SimpleRNN)")
parser.add_argument(
    "--is_augment", default=False, type=bool, help="augment data (default:False)")
args = parser.parse_args(sys.argv[1:])



# Tune of the hyperparameters =======================
project_name = args.project_name                      # The name of the SpectraPredict Model
task = args.task  					                  # 'train' / 'predict'
n_memory_steps = args.n_memory_steps                  #   time steps for encoder
n_forcast_steps = args.n_forcast_steps	    		  # time steps for decoder
train_split = args.train_split		                  # proportion as train set
batch_size = args.batch_size	                      # batch size for training
epochs = args.epochs                                  # epochs for training
test_model = args.model    		                      # 'RNN2Dense' / 'Seq2Seq_1' / 'Seq2Seq_2'
cell = args.cell           	    	                  # 'SimpleRNN' / 'LSTM' / 'GRU'
is_augment = False          		                  # True / False,  if True, x^2, x^3 will be included
# =======================================

#----------sample Dataset
#df = pd.read_csv('clean_data.csv', index_col=0, header=0)
#df.info()
#values = df.values
#------------------------

#---------Spectra Dataset-----------------
spectra = pd.read_table('PCA_ref_shift_total.txt',  delim_whitespace=True, header=None)
energies = spectra.loc[0]
spectra = spectra.drop([0])   #drop column 0 - energies
spectra = spectra.T
values = spectra.values

# data augmentation, add x^2
if is_augment:
       values = np.hstack((np.hstack((values,values**2)), values**3))


# scale data between 0 ~ 1 for better training results
data_scaler = MinMaxScaler(feature_range=(0, 1))
scaled_values = data_scaler.fit_transform(values)

# convert to supervised learning compatible format
x_timeseries = scaled_values
y_timeseries = scaled_values[:,1].reshape(-1,1)

x_train, y_train, x_test, y_test = 	timeseries_to_supervised(x_timeseries, y_timeseries, n_memory_steps, n_forcast_steps, split = train_split)
print('\n Size of x_train, y_train, x_test, y_test:')
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)


# train model & inference
if task == 'train':
	# build model
	if test_model == 'RNN2Dense_1':
		model = RNN2Dense_1(x_train.shape[1:], y_train.shape[1:], cell, 500, (20,))
	elif test_model == 'RNN2Dense_2':
		model = RNN2Dense_2(x_train.shape[1:], y_train.shape[1:], cell, 500, (20,))
	elif test_model == 'Seq2Seq_1':
		model = Seq2Seq_1(x_train.shape[1:], y_train.shape[1:], cell, 500)
	elif test_model == 'Seq2Seq_2':
		model = Seq2Seq_2(x_train.shape[1:], y_train.shape[1:], cell, 500)
	print(model.summary())
	# compile model
	model.compile()
	# train model
	history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
	   verbose=2, validation_data=(x_test,y_test))
	# save model
	model.save(project_name)

elif task == 'predict':
	# reload model
	model = load_timeseries_model(project_name)
	# predict data
	y_pred = model.predict(x_test)
	# plot results
	for n in range(n_forcast_steps):
		plt.subplot(n_forcast_steps,1,n+1)
		plt.plot(y_test[:,n],'b', label = 'True')
		plt.plot(y_pred[:,n],'r', label = 'Predict')
		plt.legend()
		plt.show()
