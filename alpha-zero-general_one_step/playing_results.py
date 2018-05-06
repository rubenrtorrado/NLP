#from Arena_2 import Arena
from MCTS import MCTS

from othello.pytorch.NNet import NNetWrapper as NNet


from NLP.NLPGame import NLPGame

from NLP.keras_2.NNet import NNetWrapper as NNet

from nltk.translate.bleu_score import sentence_bleu

import numpy as np
from utils import *

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

g = NLPGame(100)



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
Y_seq=[]
ref=[]
for i in range(0, length-seq_length, 1):
    sequence = text[i:i + seq_length]
    ref.append([sequence])
    sequence_2 = text[i + seq_length:i + seq_length + seq_length]
    label =text[i + seq_length]
    X.append([char_to_n[char] for char in sequence])
    Y.append(char_to_n[label])
    Y_seq.append([char_to_n[char] for char in sequence_2])




# nnet players
n1 = NNet(g)
n1.load_checkpoint('./temp','best.pth.tar')
args1 = dotdict({'numMCTSSims': 50, 'cpuct':1.0})
mcts1 = MCTS(g, n1, args1)
n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))


#n2 = NNet(g)
#n2.load_checkpoint('/dev/8x50x25/','best.pth.tar')
#args2 = dotdict({'numMCTSSims': 25, 'cpuct':1.0})
#mcts2 = MCTS(g, n2, args2)
#n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))
sonete=np.array(X[99])
full_string=[]
sequence=Y_seq[99]

for _ in range(100):
    pi,v=n1.predict(sonete)
    action=np.argmax(pi)
    full_string.append(action)
    sonete=np.append(sonete,action)
    sonete=sonete[1:len(sonete)]
#arena = Arena(n1p, n1p, g,args1)
#actions=arena.playGames(2, sonete,sequence,mcts1,verbose=False)
full_character=[n_to_char[value] for value in full_string]
txt=""
for char in full_character:
    txt = txt+char
print(full_character)
print(txt)