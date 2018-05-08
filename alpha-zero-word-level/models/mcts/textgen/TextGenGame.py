from __future__ import print_function

import sys

sys.path.append('..')
from models.mcts.Game import Game
from models.mcts.textgen.TextGenBoard import Board
import numpy as np
from numba import jit

class TextMaskGame(Game):

    def __init__(self, gen_data_loader, num_vocabulary, sequence_length):
        if gen_data_loader != None:
            self.gen_data_loader = gen_data_loader
            self.gen_data_loader.reset_pointer()
        self.num_vocabulary = num_vocabulary
        self.sequence_length = sequence_length
        self.end_token = num_vocabulary-1

    def getInitBoard(self):
        # return initial board (numpy board)
        sentence = self.gen_data_loader.next_example()
        board = Board(self.sequence_length, self.end_token, sentence.tolist())
        return board

    def getBoardSize(self):
        return (self.sequence_length,)

    def getActionSize(self):
        # return number of possible actions
        return self.num_vocabulary

    def getNextState(self, board, player, action):
        """
        TODO: Select next action and apply it. We can choose between:
         - Keep current word
         - Delete word
         - Add (sample) new word: sample word by selecting one amongst
           the most probable words according to a pre-trained language model

        Return new board (state) and player after applying the action
        """
        b = Board.copy(board)
        b.tokens.append(action)
        if action == self.end_token:
            while len(b.tokens) < len(b.original):
                b.tokens.append(action)
        # print('Added action: ' + str(action))
        # print('[ tokens={}\noriginal={}\n  mask={}]'.format(b.tokens, b.original, b.mask))
        # print(b.to_string())
        return b, player


    @jit
    def getValidMoves(self, board, player, professor_forcing=False):
        """
        TODO: What are our valid moves? All moves (keep, delete, sample) are always valid?

        Should return a np.float array of size n=getActionSize(), where each position is
        either 0=invalid or 1=valid. This will be used as a mask to zero-out  the computed
        action probabilities

        :param board:
        :param player:
        :return:
        """
        # return np.ones(self.getActionSize(), dtype=np.float32)
        pos = len(board.tokens)
        if professor_forcing:
            valid_positions = np.zeros(self.getActionSize(), dtype=np.float32)
            valid_positions[board.original[pos]] = 1.0
        elif board.mask[pos] == '_':
            valid_positions = np.ones(self.getActionSize(), dtype=np.float32)
        else:
            valid_positions = np.zeros(self.getActionSize(), dtype=np.float32)
            valid_positions[board.original[pos]] = 1.0
        return valid_positions

    def getGameEnded(self, board, player):
        """
        NOTES: For text generation, the game ends when the token <END> is
        generated or reached max sequence length.

        """
        isEndToken = len(board.tokens) > 0 and board.tokens[-1] == self.end_token
        if isEndToken or len(board.tokens) == self.sequence_length:
        # if len(board.tokens) == len(board.original):
            return self.getScore(board, player)
        else:
            # return -1
            return 0

    def getCanonicalForm(self, board, player):
        """
        NOTES: Should return a canonical form of this board with relation to this player.
        For text generation we probably won't need this, since the state may be
        the hidden state of a RNN and it doesn't change depending of the player.

        """
        # return state if player==1, else return -state if player==-1
        return board

    def getSymmetries(self, board, pi):
        return [(board, pi)] # no symmetries

    def stringRepresentation(self, board):
        """
        This string represents the board and is used as a key in a dictionary to store
        statistics associated to this board.

        """
        return board.to_string()

    def getScore(self, board, player):
        count = board.correct_tokens()
        # score = count / float(board.mask_size)
        score = board.correct_sentence()
        print('generated: {}'.format(board.tokens))
        print('  correct: {}'.format(count))
        print('    score: {}'.format(score))
        return score

def display(board):
    """
    TODO: Convert state (word embbeddings?) to actual words and print it.
    """
    print(" -----------------------")
    print(board.to_string)
    print(" -----------------------")
