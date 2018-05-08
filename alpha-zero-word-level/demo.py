
from models.mcts.MCTS import MCTS
from models.mcts.textgen.TextGenGame import TextMaskGame
from models.mcts.textgen.TextGenBoard import Board
# from othello.OthelloPlayers import *
from models.mcts.textgen.keras.NNet import NNetWrapper as NNet

import numpy as np
from models.mcts.utils import dotdict
import nltk

from utils.text_process import Vocabulary
from utils.text_process import text_precess, text_to_code
from utils.text_process import get_tokenlized, get_word_list, get_dict
from models.mcts.MctsDataLoader import DataLoader

data_loc = '/Users/aeciosantos/workspace/nyu/texygen/data/image_coco.txt'
# data_loc = '/Users/aeciosantos/workspace/nyu/texygen/data/testdata/test_coco.txt'
sequence_length, vocab_size = text_precess(data_loc)

tokens = get_tokenlized(data_loc)
word_set = get_word_list(tokens)
vocabulary = Vocabulary.from_word_set(word_set, sequence_length, vocab_size)


end_token = vocab_size - 1
print((sequence_length, vocab_size))



# a = input()
a = 'a deep'
text = nltk.word_tokenize(a.lower())


print(text)


codes = vocabulary.text_to_code([text])
codes = codes.strip().split(' ')

mask = [int(i) for i in codes]
for i in range(len(text), len(codes)):
    mask[i] = '_'

print(mask)

board = Board(sequence_length, end_token, mask=mask)

g = TextMaskGame(gen_data_loader=None, num_vocabulary=vocab_size,
                 sequence_length=sequence_length)


# nnet players
n1 = NNet(g)
n1.load_checkpoint('./temp/','best.pth.tar')
args1 = dotdict({'numMCTSSims': 50, 'cpuct':1.0})
mcts1 = MCTS(g, n1, args1)
player = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))


it = 0
playerNumber = 1
score = g.getGameEnded(board, playerNumber)
while score == -1:  # FIXME
    it += 1
    if True:
        print("Move ", str(it), "Player ", str(playerNumber))
        board.to_string()

    action = player(board)
    valids = g.getValidMoves(board, 2)

    if valids[action] == 0:
        print(action)
        assert valids[action] > 0
    board, _ = g.getNextState(board, playerNumber, action)
    score = g.getGameEnded(board, playerNumber)