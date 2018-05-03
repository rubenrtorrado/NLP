import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import RNN
from keras.utils import np_utils

from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.utils import *
import os


text = (open("sonnets.txt").read())
text=text.lower()

characters = sorted(list(set(text)))

n_to_char = {n:char for n, char in enumerate(characters)}
char_to_n = {char:n for n, char in enumerate(characters)}
trainning_data_p=1
X = []
Y = []
length = round(len(text)*trainning_data_p)
seq_length = 100

for i in range(0, length-seq_length, 1):
    sequence = text[i:i + seq_length]
    label =text[i + seq_length]
    X.append([char_to_n[char] for char in sequence])
    Y.append(char_to_n[label])

X_modified = np.reshape(X, (len(X), 1,seq_length))
X_modified = X_modified / float(len(characters))
Y_modified = np_utils.to_categorical(Y)

Y_modified=Y_modified[:, np.newaxis]

target_vs=np.ones((Y_modified.shape[0],1,1))

#target_vs.shape


input_boards = Input(shape=(X_modified.shape[1], X_modified.shape[2]))
extract1 = LSTM(700, return_sequences=True)(input_boards)
drop1 = Dropout(0.2)(extract1)
extract2 = LSTM(700, return_sequences=True)(drop1)
drop2=Dropout(0.2)(extract2)
extract3 = LSTM(700, return_sequences=True)(drop2)
drop3=Dropout(0.2)(extract3)
pi=Dense(38, activation='softmax', name='pi')(drop3)
v=Dense(1, activation='tanh',name='v')(drop3)
model = Model(inputs=input_boards, outputs=[pi, v])
model.compile(loss=['categorical_crossentropy', 'mean_squared_error'], optimizer='adam')

#model.fit(X_modified, Y_final, epochs=100, batch_size=50)
model.fit(x = X_modified, y = [Y_modified, target_vs], batch_size = 50, epochs = 100)
model.save_weights('text_generator_400_0.2_400_0_Alpha.2_100.h5')


def save_checkpoint(folder='checkpoint', filename='checkpoint.pth.tar'):  # done
    filepath = os.path.join(folder, filename)
    if not os.path.exists(folder):
        print("Checkpoint Directory does not exist! Making directory {}".format(folder))
        os.mkdir(folder)
    else:
        print("Checkpoint Directory exists! ")
    model.save_weights(filepath)


def load_checkpoint(folder='checkpoint', filename='checkpoint.pth.tar'):  # done
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
    filepath = os.path.join(folder, filename)
    if not os.path.exists(filepath):
        raise ("No model in path {}".format(filepath))
    model.load_weights(filepath)


#save_checkpoint(folder='./temp/', filename='temp.pth.tar')
#save_checkpoint(folder='./temp/', filename='checkpoint')
#save_checkpoint(folder='./temp/', filename='best.pth.tar')

model.load_weights('text_generator_400_0.2_400_0_Alpha.2_100.h5')

string_mapped = np.reshape(X[200], (1, 1,seq_length))
full_string=string_mapped
print(string_mapped)
print(np.shape(string_mapped))

#full_string = [n_to_char[value] for value in string_mapped]
# generating characters
for i in range(400):
    #x = np.reshape(string_mapped,(1,len(string_mapped), 1))
    #x = x / float(len(characters))
    x=string_mapped
    x = x / float(len(characters))

    pred_index = np.argmax(model.predict(x, verbose=0)[0])
    seq=pred_index
    #seq = [n_to_char[value] for value in string_mapped]
    full_string=np.append(full_string,pred_index)

    string_mapped=np.append(string_mapped,pred_index)
    string_mapped = string_mapped[1:len(string_mapped)]
    string_mapped = np.reshape(string_mapped, (1, 1, seq_length))

full_character=[n_to_char[value] for value in full_string]
txt=""
for char in full_character:
    txt = txt+char
print(full_character)
print(txt)


