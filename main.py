import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from SpectraTimeSeries.SeriesModels.RNN2Dense import *
from SpectraTimeSeries.SeriesModels.Seq2Seq import *
from SpectraTimeSeries.Utils.utils import timeseries_to_supervised,load_timeseries_model
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
from cesium import datasets
from cesium import featurize
import seaborn as sns

parser = argparse.ArgumentParser(description="SpectraPredict Framework")
parser.add_argument("--project_name", default="SpectraPredict", type=str, help="Name of SpectraPredict Model (default:SpectaPredict")
parser.add_argument("--spectra_data", default="data/PCA_ref_shift_total.txt", type=str, help="Data file name (default: data/PCA_ref_shift_total.txt")
parser.add_argument("--task", default="plot", type=str, help="train/predict/plot (default:train)" )
parser.add_argument("--n_memory_steps",  default=5, type=int, help="number of steps >=1 (default:1)")
parser.add_argument("--n_forecast_steps", default=5, type=int, help="number of forecast steps >=1 (default:1)")
parser.add_argument("--train_split", default=0.1, type=float, help=" train / split ratio from 0.1 to 0.9 (default:0.2)")
parser.add_argument("--batch_size", default=1, type=int, help="batch size (default:1)")
parser.add_argument("--epochs", default=1, type=int, help="epochs >=1 (default: 1)")
parser.add_argument("--model", default="RNN2Dense_2", type=str, help="time series model RNN2Dense_2/Seq2Seq_1/Seq2Seq_2 (default:RNN2Dense_2)")
parser.add_argument("--cell", default="SimpleRNN",  type=str,  help="type of cell: SimpleRNN/LSTM/GRU (default:SimpleRNN)")
parser.add_argument("--is_augment", default=False, type=bool, help="augment data (default:False)")
args = parser.parse_args(sys.argv[1:])


project_name = args.project_name
task = args.task
spectra_data = args.spectra_data
n_memory_steps = args.n_memory_steps
n_forecast_steps = args.n_forecast_steps
train_split = args.train_split
batch_size = args.batch_size
epochs = args.epochs
test_model = args.model
cell = args.cell
is_augment = False
predictions = []

#---------Read Spectra Dataset-----------------
spectra = pd.read_table(spectra_data,  delim_whitespace=True, header=None)
print("Spectra File {} loaded sucessfully".format(spectra_data))
row, col = spectra.shape
print("Loaded spectra has {} points per one spectra and {} measured spectra ".format(col, row))
energies = spectra.loc[0]
spectra = spectra.drop([0])
print("Droping energies, spectra  has {} points per one spectra and {} measured spectra ".format(col, row))
print(spectra.shape)
spectra = spectra.T
values = spectra.values

#if augmented
if is_augment:
       values = np.hstack((np.hstack((values,values**2)), values**3))

# normalizing spectra
#data_scaler = MinMaxScaler(feature_range=(0, 1))
#scaled_values = data_scaler.fit_transform(values)

#------------------Spectra preprocesing------------------------------------------------------
#x_timeseries = scaled_values
#y_timeseries = scaled_values[:,1].reshape(-1,1)

x_timeseries = values
y_timeseries = values[:,1].reshape(-1,1)

print ('X_timeSeries', x_timeseries.shape)
print ('Y_timeSeries', y_timeseries.shape)

x_train, y_train, x_test, y_test = 	timeseries_to_supervised(x_timeseries, y_timeseries, n_memory_steps, n_forecast_steps, split = train_split)

#------------------Model trainig and save-------------------------------------------------
if task == 'train':
	if test_model == 'RNN2Dense_1':
		model = RNN2Dense_1(x_train.shape[1:], y_train.shape[1:], cell, col, (20,))
	elif test_model == 'RNN2Dense_2':
		model = RNN2Dense_2(x_train.shape[1:], y_train.shape[1:], cell, col, (20,))
	elif test_model == 'Seq2Seq_1':
		model = Seq2Seq_1(x_train.shape[1:], y_train.shape[1:], cell, col)
	elif test_model == 'Seq2Seq_2':
		model = Seq2Seq_2(x_train.shape[1:], y_train.shape[1:], cell, col)
	print("Model is sucessfully created")
	print(model.summary())
	model.compile()
	history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,verbose=2, validation_data=(x_test,y_test))
	model.save('data/'+project_name)
	print('Model is saved into file {}'.format(project_name))

#-----------------Spectra prediction and save--------------------------------------------
elif task == 'predict':
	model = load_timeseries_model('data/'+project_name)
	print('Model is loaded from file {}'.format(project_name))
	y_pred = model.predict(x_test)
	predictions = y_pred
	d1, d2, d3 = predictions.shape
	print(predictions.shape)
	print("Predicted spectra has {} measured spectra  with {} forecast steps and {} points per one measurement ".format(d1, d2, d3))
	save_pred = predictions[:,0,:]
	df = pd.DataFrame(save_pred)
	df.to_csv('data/'+project_name+'.csv')


#-----------------Spectra plot----------------------------------------------------------
elif task == 'plot':
	model = load_timeseries_model('data/'+project_name)
	print('Model is loaded from file {}'.format(project_name))
	print('X_test dimensions', x_test.shape)
	y_pred = model.predict(x_test)
	predictions = y_pred
	d1, d2, d3 = predictions.shape
	print("Predicted spectra has {} measured spectra  with {} forecast steps and {} points per one measurement ".format(d1, d2, d3))
	arr = predictions[:,0,:]
	for n in range(n_forecast_steps):
		plt.subplot(n_forecast_steps,1,n+1)
		plt.plot(y_test[:,n],'b', label = 'True')
		plt.plot(arr,'r', label = 'Predict')
		plt.legend()
		plt.show()
