from collections import deque
from Arena_demo_Bleu import Arena
from MCTS_Bleu import MCTS
import numpy as np
from pytorch_classification.utils import Bar, AverageMeter
import time, os, sys
from pickle import Pickler, Unpickler
from random import shuffle
from keras.utils import np_utils

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

import nltk

class Coach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """
    def __init__(self,player1,player2, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game)  # the competitor network
        self.args = args
        self.mcts = MCTS(self.game, self.nnet, self.args)
        self.trainExamplesHistory = []    # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False # can be overriden in loadTrainExamples()
        self.player1=player1
        self.player2=player2
    def executeEpisode(self):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.

        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.

        Returns:
            trainExamples: a list of examples of the form (canonicalBoard,pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        trainExamples = []
        board = self.game.getInitBoard()
        self.curPlayer = 1
        episodeStep = 0

        while True:
            episodeStep += 1
            canonicalBoard = self.game.getCanonicalForm(board,self.curPlayer)
            temp = int(episodeStep < self.args.tempThreshold)

            pi = self.mcts.getActionProb(canonicalBoard, temp=temp)
            sym = self.game.getSymmetries(canonicalBoard, pi)
            for b,p in sym:
                trainExamples.append([b, self.curPlayer, p, None])

            action = np.random.choice(len(pi), p=pi)
            board, self.curPlayer = self.game.getNextState(board, self.curPlayer, action)

            r = self.game.getGameEnded(board, self.curPlayer)

            if r!=0:
                return [(x[0],x[2],r*((-1)**(x[1]!=self.curPlayer))) for x in trainExamples]

    def playGame(self, target_sentence,reference,n_to_char,verbose=False):
        """
        Executes one episode of a game.

        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """
        players = [self.player2, None, self.player1]
        curPlayer = 1
        board = self.game.getInitBoard_BLEU(target_sentence,reference,n_to_char)
        it = 0
        actions=[]
        while self.game.getGameEnded_BLEU(board, curPlayer)==0:
            it+=1
            if verbose:
                assert(self.display)
                print("Turn ", str(it), "Player ", str(curPlayer))
                self.display(board)
            action = players[curPlayer+1](self.game.getCanonicalForm(board, curPlayer))
            actions.append(action)
            valids = self.game.getValidMoves(self.game.getCanonicalForm(board, curPlayer),1)

            if valids[action]==0:
                print(action)
                assert valids[action] >0
            board, curPlayer = self.game.getNextState(board, curPlayer, action)
        if verbose:
            assert(self.display)
            print("Game over: Turn ", str(it), "Result ", str(self.game.getGameEnded(board, 1)))
            self.display(board)
        return self.game.getGameEnded(board, 1),actions

    def test(self,target_sonnets,verbose=False):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximium length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """
        self.n = 100
        text = (open("data/sonnets.txt").read())
        text = text.lower()

        characters = sorted(list(set(text)))

        n_to_char = {n: char for n, char in enumerate(characters)}
        char_to_n = {char: n for n, char in enumerate(characters)}

        X = []
        Y = []
        Y_seq=[]
        length = len(text)
        seq_length = 100

        for i in range(0, length - seq_length, 1):
            sequence = text[i:i + seq_length]
            sequence_2 = text[i + seq_length:i + seq_length + seq_length]
            label = text[i + seq_length]
            X.append([char_to_n[char] for char in sequence])
            Y_seq.append([char_to_n[char] for char in sequence_2])
            Y.append(char_to_n[label])
        self.X=X
        self.Y=Y
        #self.Y_seq=Y_seq
        X_modified = np.reshape(X, (len(X), seq_length, 1))
        self.X_modified = X_modified / float(len(characters))
        self.Y_modified = np_utils.to_categorical(Y)
        #self.nround=0
        self.Y_seq = Y
        self.Y_seq_target=Y_seq
        n_sonete=target_sonnets
        target_sentence=X[n_sonete]
        target_y=Y[n_sonete]
        #model=self.Load_LSTM()
        #target_y = self.LSTM(target_sentence,model)

        reference = nltk.word_tokenize(text)

        actions=[]

        eps_time = AverageMeter()
        bar = Bar('Arena.playGames', max=100)
        end = time.time()
        eps = 0
        maxeps = int(100)
        finalScore1=0
        finalScore2=0

        num = int(100/2)
        oneWon = 0
        twoWon = 0
        draws = 0
        gameResults=[]


        #self.player1, self.player2 = self.player1, self.player1




        for _ in range(100):
            gameResult,actions = self.playGame(target_sentence,reference,n_to_char,verbose=verbose)


    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration)+".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed

    def loadTrainExamples(self):
        modelFile = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
        examplesFile = modelFile+".examples"
        if not os.path.isfile(examplesFile):
            print(examplesFile)
            r = input("File with trainExamples not found. Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            print("File with trainExamples found. Read it.")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            f.closed
            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True

    def LSTM(self,x,model):



        x = np.array(x) / 100
        np.reshape(x, (1, 1, 100))
        #x = x / float(100)
        x = x[np.newaxis, :]
        x = x[np.newaxis, :]

        pred_index = np.argmax(model.predict(x, verbose=0)[0])

        return pred_index

    def Load_LSTM(selfself):
        input_boards = Input(shape=(1, 100))
        extract1 = LSTM(700, return_sequences=True)(input_boards)
        drop1 = Dropout(0.2)(extract1)
        extract2 = LSTM(700, return_sequences=True)(drop1)
        drop2 = Dropout(0.2)(extract2)
        extract3 = LSTM(700, return_sequences=True)(drop2)
        drop3 = Dropout(0.2)(extract3)
        pi = Dense(38, activation='softmax', name='pi')(drop3)
        v = Dense(1, activation='tanh', name='v')(drop3)
        model = Model(inputs=input_boards, outputs=[pi, v])
        model.compile(loss=['categorical_crossentropy', 'mean_squared_error'], optimizer='adam')

        model.load_weights('text_generator_400_0.2_400_0_Alpha.2_100.h5')

        return model


