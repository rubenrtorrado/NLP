from __future__ import print_function

import sys

sys.path.append('..')
from models.mcts.Game import Game
from models.mcts.textgen.TextGenBoard import Board
import numpy as np
from numba import jit

class CompleteNgramGame(Game):

    def __init__(self, gen_data_loader, num_vocabulary, sequence_length):
        self.num_vocabulary = num_vocabulary
        self.gen_data_loader = gen_data_loader
        self.gen_data_loader.reset_pointer()
        self.sequence_length = sequence_length
        self.end_token = num_vocabulary-1

        eof = False
        while(not eof):
            sentence, eof = self.gen_data_loader.next_example(return_eof=True)
            print('Sentence: ', sentence)
            for i in range(0, len(sentence)-4):
                print(sentence[i:i + 2])
                print(sentence[i:i + 3])
                print(sentence[i:i + 4])
                # print(self.end_token)
                if sentence[i+3] == self.end_token:
                    break




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
        # print('Added action: ' + str(action))
        # print('[ tokens={}\noriginal={}\n  mask={}]'.format(b.tokens, b.original, b.mask))
        # print(b.to_string())
        return b, player
        # if player takes action on board, return next (board,player)
        # action must be a valid move
        # if action == self.n*self.n:
        #     return (board, -player)
        # b = Board(self.n)
        # b.pieces = np.copy(board)
        # move = (int(action/self.n), action%self.n)
        # b.execute_move(move, player)
        # return (b.pieces, -player)


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
        if len(board.tokens) == len(board.original):
            return board.correct_tokens()
        else:
            return -1

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
        """
        TODO: Decide which metrics to use to compute score.
              Use BLEU for sentence editing, and perplexity for language generation?
        """
        return board.correct_tokens()

def display(board):
    """
    TODO: Convert state (word embbeddings?) to actual words and print it.
    """
    print(" -----------------------")
    print(board.to_string)
    print(" -----------------------")


if __name__ == '__main__':
    from utils.text_process import text_precess, text_to_code
    from utils.text_process import get_tokenlized, get_word_list, get_dict
    from models.mcts.MctsDataLoader import DataLoader

    a = {
        tuple([0,1]): 1
    }
    print(tuple([0,1]) in a)

    data_loc = '/Users/aeciosantos/workspace/nyu/texygen/data/image_coco.txt'
    sequence_length, vocab_size = text_precess(data_loc)
    gen_dataloader = DataLoader(batch_size=64, seq_length=sequence_length)
    gen_dataloader.create_batches('save/oracle.txt')

    g = CompleteNgramGame(gen_data_loader=gen_dataloader, num_vocabulary=vocab_size,
                          sequence_length=sequence_length)


    tokens = get_tokenlized(data_loc)
    word_set = get_word_list(tokens)

    [word_index_dict, index_word_dict] = get_dict(word_set)

