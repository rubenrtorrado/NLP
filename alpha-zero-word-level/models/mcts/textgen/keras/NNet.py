import os
import sys
import time

import numpy as np

sys.path.append('..')
from models.mcts.NeuralNet import NeuralNet

from .MctsNNet import MctsNNet as onnet
from models.mcts.utils import *
import keras


args = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 50,
    'batch_size': 128,
    'cuda': True,
    'num_channels': 512,
})

class NNetWrapper(NeuralNet):
    def __init__(self, game):
        self.nnet = onnet(game, args)
        self.board_dim = game.getBoardSize()
        self.action_size = game.getActionSize()

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        input_boards, target_pis, target_vs = list(zip(*examples))
        input_boards = np.asarray(input_boards)
        target_pis = np.asarray(target_pis)
        target_vs = np.asarray(target_vs)


        print('TARGET Vs:', np.histogram(target_vs))
        for i in range(len(input_boards)):
            print(target_vs[i], input_boards[i])

        tb_callback = keras.callbacks.TensorBoard(log_dir='./tensorboard', histogram_freq=0,
                                                  write_graph=True, write_images=True)

        self.nnet.model.fit(x = input_boards, y = [target_pis, target_vs],
                            batch_size = args.batch_size, epochs = args.epochs,
                            callbacks=[tb_callback])
        

    def predict(self, board):
        """
        board: np array with board
        """
        # timing
        # start = time.time()

        # preparing input
        #board = board[np.newaxis, :, :]
        # run
        # pi, v = self.nnet.model.predict(board)

        x = board.getModelInput()
        x = x[np.newaxis, :]
        # print(x.shape)
        # print(x)
        pi, v = self.nnet.model.predict(x)
        #print('predictions', pi.shape, v.shape)

        #print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return pi[0], v[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        self.nnet.model.save_weights(filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise("No model in path {}".format(filepath))
        self.nnet.model.load_weights(filepath)
