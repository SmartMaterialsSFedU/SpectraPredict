import numpy as np
import pickle 
from SpectraTimeSeries.SeriesModels.RNN2Dense import *
from SpectraTimeSeries.SeriesModels.Seq2Seq import *
from keras.models import load_model

def timeseries_to_supervised(x_timeseries, y_timeseries, n_memory_step, n_forcast_step, split = None):

    assert len(x_timeseries.shape) == 2, 'x_timeseries must be shape of (time_step, features)'
    assert len(y_timeseries.shape) == 2, 'y_timeseries must be shape of (time_step, features)' 

    input_step, input_feature = x_timeseries.shape
    output_step, output_feature = y_timeseries.shape
    assert input_step == output_step, 'number of time_step of x_timeseries and y_timeseries are not consistent!'

    n_RNN_sample=input_step-n_forcast_step-n_memory_step+1
    RNN_x=np.zeros((n_RNN_sample,n_memory_step, input_feature))
    RNN_y=np.zeros((n_RNN_sample,n_forcast_step, output_feature))

    for n in range(n_RNN_sample):
        RNN_x[n,:,:]=x_timeseries[n:n+n_memory_step,:]
        RNN_y[n,:,:]=y_timeseries[n+n_memory_step:n+n_memory_step+n_forcast_step,:]
    if split != None:
        assert (split <=0.9) & (split >= 0.1), 'Split must be from 0.1 to 0.9'
        return RNN_x[:int(split*len(RNN_x))], RNN_y[:int(split*len(RNN_x))],\
               RNN_x[int(split*len(RNN_x))+1:], RNN_y[int(split*len(RNN_x))+1:]
    else:
        return RNN_x, RNN_y, None, None


def load_timeseries_model(model_name):
    model_timeseries = load_model(model_name+'.h5')
    model_info = pickle.load(open(model_name+'.info','rb'))
    print('\nloading '+model_info['class']+' model with '+model_info['cell']+' cell ...')


    model_name = model_info['class']
    model_info.pop('class')


    if model_name == 'RNN2Dense_1':
        model = RNN2Dense_1(**model_info,reload = True)
    elif 'RNN2Dense_2':
        model = RNN2Dense_2(**model_info,reload = True)
    elif 'Seq2Seq_1':
        model = Seq2Seq_1(**model_info,reload = True)
    elif 'Seq2Seq_2':
        model = Seq2Seq_2(**model_info,reload = True)
    

    model_info['class'] = model_name

    print('Model Information:')
    print(model_info)


    model.model = model_timeseries 
    model.class_info = model_info

    return model