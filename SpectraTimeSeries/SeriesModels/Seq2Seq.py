import numpy as np
from keras.models import Model
from keras.layers import Input, SimpleRNN, Dense, GRU, LSTM, Reshape, Lambda
from keras import backend as K
from SpectraTimeSeries.SeriesModels.TimeSeriesBase import TSBase


class Seq2Seq_1(TSBase):
    def __init__(self, input_shape, output_shape, cell, cell_units, reload = False ):

        if reload:
            self.model = None
            self.class_info = None
        else:

            self.class_info = {'class': 'Seq2Seq', 'input_shape': input_shape, 'output_shape': output_shape,
                'cell': cell, 'cell_units': cell_units}

            if cell == 'LSTM':

                encoder = LSTM(units = cell_units, return_state = True)
                decoder = LSTM(units = cell_units, return_sequences=True, return_state = True)
                decoder_dense = Dense(output_shape[-1])


                encoder_input = Input(input_shape)
                encoder_output, state_h, state_c = encoder(encoder_input)
                encoder_state = [state_h, state_c]  

                decoder_input = Input((1,output_shape[-1]))
                

                iter_input = decoder_input
                iter_state = encoder_state
                all_output = []

                for _ in range(output_shape[0]):

                    output, state_h, state_c = decoder(iter_input, initial_state=iter_state)
                    output = decoder_dense(output)


                    all_output.append(output)
                    

                    iter_input = output
                    iter_state = [state_h, state_c]

            elif cell == 'SimpleRNN':

                encoder = SimpleRNN(units = cell_units, return_state = True)
                decoder = SimpleRNN(units = cell_units, return_sequences=True, return_state = True)
                decoder_dense = Dense(output_shape[-1])


                encoder_input = Input(input_shape)
                encoder_output, state_h = encoder(encoder_input)
                encoder_state = state_h  

                decoder_input = Input((1,output_shape[-1]))
                

                iter_input = decoder_input
                iter_state = encoder_state
                all_output = []

                for _ in range(output_shape[0]):

                    output, state_h = decoder(iter_input, initial_state=iter_state)
                    output = decoder_dense(output)


                    all_output.append(output)
                    

                    iter_input = output
                    iter_state = state_h

            elif cell == 'GRU':

                encoder = GRU(units = cell_units, return_state = True)
                decoder = GRU(units = cell_units, return_sequences=True, return_state = True)
                decoder_dense = Dense(output_shape[-1])


                encoder_input = Input(input_shape)
                encoder_output, state_h = encoder(encoder_input)
                encoder_state = state_h  

                decoder_input = Input((1,output_shape[-1]))
                

                iter_input = decoder_input
                iter_state = encoder_state
                all_output = []

                for _ in range(output_shape[0]):

                    output, state_h = decoder(iter_input, initial_state=iter_state)
                    output = decoder_dense(output)


                    all_output.append(output)
                    

                    iter_input = output
                    iter_state = state_h


            decoder_output = Lambda(lambda x: K.concatenate(x, axis=1))(all_output)
            self.model = Model([encoder_input, decoder_input], decoder_output)

    def data_preprocessing(self, x, y = None):
        x_new = [x, np.ones((len(x), 1, self.class_info['output_shape'][1]))]
        y_new = y

        return x_new, y_new


class Seq2Seq_2(TSBase):
    def __init__(self, input_shape, output_shape, cell, cell_units, reload = False ):

        if reload:
            self.model = None
            self.class_info = None
        else:

            self.class_info = {'class': 'Seq2Seq', 'input_shape': input_shape, 'output_shape': output_shape,
                'cell': cell, 'cell_units': cell_units}

            if cell == 'LSTM':

                encoder = LSTM(units = cell_units, return_state = True)
                decoder = LSTM(units = cell_units, return_sequences=True, return_state = True)
                decoder_dense = Dense(output_shape[-1])


                encoder_input = Input(input_shape)
                encoder_output, state_h, state_c = encoder(encoder_input)
                encoder_state = [state_h, state_c]  
                

                iter_input = Reshape((1,output_shape[-1]))(decoder_dense(state_h))
                iter_state = encoder_state
                all_output = []

                for _ in range(output_shape[0]):

                    output, state_h, state_c = decoder(iter_input, initial_state=iter_state)
                    output = decoder_dense(output)


                    all_output.append(output)
                    

                    iter_input = output
                    iter_state = [state_h, state_c]

            elif cell == 'SimpleRNN':

                encoder = SimpleRNN(units = cell_units, return_state = True)
                decoder = SimpleRNN(units = cell_units, return_sequences=True, return_state = True)
                decoder_dense = Dense(output_shape[-1])


                encoder_input = Input(input_shape)
                encoder_output, state_h = encoder(encoder_input)
                encoder_state = state_h  
                

                iter_input = Reshape((1,output_shape[-1]))(decoder_dense(state_h))
                iter_state = encoder_state
                all_output = []

                for _ in range(output_shape[0]):

                    output, state_h = decoder(iter_input, initial_state=iter_state)
                    output = decoder_dense(output)


                    all_output.append(output)
                    
                    iter_input = output
                    iter_state = state_h

            elif cell == 'GRU':

                encoder = GRU(units = cell_units, return_state = True)
                decoder = GRU(units = cell_units, return_sequences=True, return_state = True)
                decoder_dense = Dense(output_shape[-1])


                encoder_input = Input(input_shape)
                encoder_output, state_h = encoder(encoder_input)
                encoder_state = state_h  
                

                iter_input = Reshape((1,output_shape[-1]))(decoder_dense(state_h))
                iter_state = encoder_state
                all_output = []

                for _ in range(output_shape[0]):

                    output, state_h = decoder(iter_input, initial_state=iter_state)
                    output = decoder_dense(output)


                    all_output.append(output)
                    

                    iter_input = output
                    iter_state = state_h

            decoder_output = Lambda(lambda x: K.concatenate(x, axis=1))(all_output)
            self.model = Model(encoder_input, decoder_output)