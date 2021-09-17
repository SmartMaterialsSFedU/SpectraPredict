import numpy as np
from keras.models import Model
from keras.layers import Input, SimpleRNN, Dense, GRU, LSTM, Dropout, Reshape, Lambda
from keras import backend as K
from SpectraTimeSeries.SeriesModels.TimeSeriesBase import TSBase

class RNN2Dense_1(TSBase):
    def __init__(self, input_shape, output_shape, cell, cell_units, 
        dense_units = None, dropout_rate = 0.3, reload = False ):


        if reload:
            self.model = None
            self.class_info = None
        else:

            self.class_info={'class': 'RNN2Dense_1', 'input_shape': input_shape, 
                'output_shape': output_shape, 'cell': cell, 'cell_units': cell_units,
                'dense_units': dense_units} 


            assert cell in ['SimpleRNN', 'LSTM', 'GRU']
            assert type(cell_units) == int
            assert type(dense_units) == tuple
            

            x_in = Input(input_shape)
            if cell == 'SimpleRNN':
                x = SimpleRNN(units=cell_units)(x_in)
            elif cell == 'LSTM':
                x = LSTM(units=cell_units)(x_in)
            elif cell == 'GRU':
                x = GRU(units=cell_units)(x_in)
            if dense_units != None:
                for n_units in dense_units:
                    x = Dense(n_units, activation='relu')(x)
                    x = Dropout(dropout_rate)(x)
            x = Dense(np.prod(output_shape))(x)
            x_out = Reshape((output_shape))(x)
            self.model = Model(inputs = x_in, outputs = x_out)



class RNN2Dense_2(TSBase):
    def __init__(self, input_shape, output_shape, cell, cell_units, 
        dense_units = None, dropout_rate = 0.3, reload = False ):


        if reload:
            self.model = None
            self.class_info = None
        else:

            self.class_info={'class': 'RNN2Dense_2', 'input_shape': input_shape, 
                'output_shape': output_shape, 'cell': cell, 'cell_units': cell_units,
                'dense_units': dense_units} 


            assert cell in ['SimpleRNN', 'LSTM', 'GRU']
            assert type(cell_units) == int
            assert type(dense_units) == tuple
            

            x_in = Input(input_shape)
            if cell == 'SimpleRNN':
                x = SimpleRNN(units=cell_units, return_sequences=True)(x_in)
            elif cell == 'LSTM':
                x = LSTM(units=cell_units, return_sequences=True)(x_in)
            elif cell == 'GRU':
                x = GRU(units=cell_units, return_sequences=True)(x_in)

            self.model = Model(inputs = x_in, outputs = x)