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

text = (open("sonnets.txt").read())
text=text.lower()

characters = sorted(list(set(text)))

n_to_char = {n:char for n, char in enumerate(characters)}
char_to_n = {char:n for n, char in enumerate(characters)}

X = []
Y = []
length = len(text)
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

model.load_weights('text_generator_400_0.2_400_0_Alpha.2_100.h5')

string_mapped = X[99]
full_string = [n_to_char[value] for value in string_mapped]
# generating characters
for i in range(400):
    x = np.reshape(string_mapped,(1,len(string_mapped), 1))
    x = x / float(len(characters))

    pred_index = np.argmax(model.predict(x, verbose=0))
    seq = [n_to_char[value] for value in string_mapped]
    full_string.append(n_to_char[pred_index])

    string_mapped.append(pred_index)
    string_mapped = string_mapped[1:len(string_mapped)]

txt=""
for char in full_string:
    txt = txt+char
print(txt)
