import Arena
from MCTS import MCTS
from models.mcts.textgen.TextGenGame import TextMaskGame
from models.mcts.textgen.TextGenBoard import Board
# from othello.OthelloPlayers import *
# from othello.pytorch.NNet import NNetWrapper as NNet

import numpy as np
from utils import *

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""
from ..utils.text_process import Vocabulary
from utils.text_process import text_precess, text_to_code
from utils.text_process import get_tokenlized, get_word_list, get_dict
from models.mcts.MctsDataLoader import DataLoader

data_loc = '/Users/aeciosantos/workspace/nyu/texygen/data/image_coco.txt'
sequence_length, vocab_size = text_precess(data_loc)

tokens = get_tokenlized(data_loc)
word_set = get_word_list(tokens)
vocabulary = Vocabulary.from_word_set(word_set, sequence_length, vocab_size)
vocabulary.tokens_to_codes_file(tokens, data_loc)

end_token = vocab_size - 1
print((sequence_length, vocab_size))



a = input()
tokens = [x for x in a.split(' ')]
codes = vocabulary.text_to_code(tokens)
print(codes)



board = Board(sequence_length, end_token, codes.tolist())

#
#
#
# # g = OthelloGame(6)
# g = TextMaskGame()
# # all players
# rp = RandomPlayer(g).play
# # gp = GreedyOthelloPlayer(g).play
# hp = HumanOthelloPlayer(g).play
#
# # nnet players
# n1 = NNet(g)
# n1.load_checkpoint('./pretrained_models/othello/pytorch/','6x100x25_best.pth.tar')
# args1 = dotdict({'numMCTSSims': 50, 'cpuct':1.0})
# mcts1 = MCTS(g, n1, args1)
# n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))
#
#
# #n2 = NNet(g)
#n2.load_checkpoint('/dev/8x50x25/','best.pth.tar')
#args2 = dotdict({'numMCTSSims': 25, 'cpuct':1.0})
#mcts2 = MCTS(g, n2, args2)
#n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))
#
# arena = Arena.Arena(n1p, hp, g, display=display)
# print(arena.playGames(2, verbose=True))
