import sys
sys.path.append('..')
from utils import *

import argparse
from keras.models import *
from keras.layers import *
from keras.optimizers import *

#from keras.models import Sequential
#from keras.layers import Dense
#from keras.layers import Dropout
#from keras.layers import LSTM
#from keras.layers import RNN
from keras.utils import np_utils

class NLPNNet():
    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y = game.getBoardSize()#done
        self.action_size = game.getActionSize()#done
        self.args = args#done

        # Neural Net
        #self.input_boards = Input(shape=(self.board_x, self.board_y))    # s: batch_size x board_x x board_y

        #x_image = Reshape((self.board_x, self.board_y, 1))(self.input_boards)                # batch_size  x board_x x board_y x 1
        #h_conv1 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(args.num_channels, 3, padding='same')(x_image)))         # batch_size  x board_x x board_y x num_channels
        #h_conv2 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(args.num_channels, 3, padding='same')(h_conv1)))         # batch_size  x board_x x board_y x num_channels
        #h_conv3 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(args.num_channels, 3, padding='valid')(h_conv2)))        # batch_size  x (board_x-2) x (board_y-2) x num_channels
        #h_conv4 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(args.num_channels, 3, padding='valid')(h_conv3)))        # batch_size  x (board_x-4) x (board_y-4) x num_channels
        #h_conv4_flat = Flatten()(h_conv4)
        #s_fc1 = Dropout(args.dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(1024)(h_conv4_flat))))  # batch_size x 1024
        #s_fc2 = Dropout(args.dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(512)(s_fc1))))          # batch_size x 1024
        #self.pi = Dense(self.action_size, activation='softmax', name='pi')(s_fc2)   # batch_size x self.action_size
        #self.v = Dense(1, activation='tanh', name='v')(s_fc2)                    # batch_size x 1

        #self.model = Model(inputs=self.input_boards, outputs=[self.pi, self.v])
        #self.model.compile(loss=['categorical_crossentropy','mean_squared_error'], optimizer=Adam(args.lr))

        #NNT architecture

        #self.model = Sequential()
        #self.model.add(LSTM(700, input_shape=(X_modified.shape[1], X_modified.shape[2]), return_sequences=True))
        #self.model.add(LSTM(700, input_shape=(self.board_x, self.board_y), return_sequences=True))
        #self.model.add(Dropout(0.2))
        #self.model.add(LSTM(700))
        #self.model.add(Dropout(0.2))
        #self.model.add(Dense(Y_modified.shape[1], activation='softmax'))
        #self.model.add(Dense(self.action_size, activation='softmax',name='pi'))
        #self.model.add(Dense(1, activation='tanh',name='v'))
        #self.model.compile(loss='categorical_crossentropy', optimizer=Adam(args.lr))
        #self.model.compile(loss=['categorical_crossentropy','mean_squared_error'], optimizer=Adam(args.lr))
        #self.input_boards = Input(shape=(self.board_x, self.board_y))
        self.input_boards = Input(shape=(self.board_y, self.board_x))
        #x_image = Reshape((self.board_x, self.board_y, 1))(self.input_boards)
        extract1 = GRU(10, return_sequences=True)(self.input_boards)
        drop1 = Dropout(0.2)(extract1)
        extract2 = GRU(10, return_sequences=True)(drop1)
        #drop2=Dropout(0.2)(extract2)
        #extract3=LSTM(400, return_sequences=True)(drop2)
        #drop3 = Dropout(0.2)(extract3)
        self.pi=Dense(self.action_size, activation='softmax', name='pi')(extract2)
        self.v=Dense(1, activation='tanh',name='v')(extract2)
        self.model = Model(inputs=self.input_boards, outputs=[self.pi, self.v])
        self.model.compile(loss=['categorical_crossentropy', 'mean_squared_error'], optimizer=Adam(args.lr))
