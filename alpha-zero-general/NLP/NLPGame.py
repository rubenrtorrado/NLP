from __future__ import print_function
import sys
sys.path.append('..')
from Game import Game
from .OthelloLogic import Board
import numpy as np
from keras.utils import np_utils


class NLPGame(Game):
    def __init__(self, n):#done
        #n is the sequence of words
        self.n = n
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
        self.Y_seq=Y_seq
        X_modified = np.reshape(X, (len(X), seq_length, 1))
        self.X_modified = X_modified / float(len(characters))
        self.Y_modified = np_utils.to_categorical(Y)
        self.nround=0

    def getInitBoard(self):#done
        # return initial board (numpy board)
        #sample of encoder and decoder
        #b = Board(self.n)
        #return np.array(b.pieces)
        # sample of encoder and decoder
        self.nround=0
        r=np.random.randint(np.shape(self.X)[0], size=1)
        return self.X[r]

    def getBoardSize(self):#done
        # (a,b) tuple
        #return (self.n, self.n)
        return (self.X_modified.shape[1],self.X_modified.shape[2])

    def getActionSize(self):#done
        # return number of actions
        #return self.n*self.n + 1
        return self.Y_modified.shape[1]+1

    def getNextState(self, board, player, action):#done the player should be 1 or -1
        # if player takes action on board, return next (board,player)
        # action must be a valid move
        #if action == self.n*self.n:
        #    return (board, -player)
        #b = Board(self.n)
        #b.pieces = np.copy(board)
        #move = (int(action/self.n), action%self.n)
        #b.execute_move(move, player)
        #return (b.pieces, -player)
        self.nround += self.nround
        board.pop(0)
        board=board.append(action)
        return (board,-player)

    def getValidMoves(self, board, player):#done
        # return a fixed size binary vector
        #valids = [0]*self.getActionSize()
        #b = Board(self.n)
        #b.pieces = np.copy(board)
        #legalMoves =  b.get_legal_moves(player)
        #if len(legalMoves)==0:
        #    valids[-1]=1
        #    return np.array(valids)
        #for x, y in legalMoves:
        #    valids[self.n*x+y]=1
        return np.array(self.Y_modified.shape[1])

    def getGameEnded(self, board, player):#REVIEW
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        # player = 1


        #b = Board(self.n)
        #b.pieces = np.copy(board)
        #if b.has_legal_moves(player):
        #    return 0
        #if b.has_legal_moves(-player):
        #    return 0
        #if b.countDiff(player) > 0:
        #    return 1
        #return -1

        #if the game is cooperative, we dont need more -1

        # I think the score should the final score of the metric
        if self.nround <=self.n:
            return 0
        else:
            return 1

    def getCanonicalForm(self, board, player):#done
        # return state if player==1, else return -state if player==-1
        #return player*board
        return board
    def getSymmetries(self, board, pi):#done # dimension (state, pi (36 moves+v))
        # mirror, rotational
    #    assert(len(pi) == self.n**2+1)  # 1 for pass
    #    pi_board = np.reshape(pi[:-1], (self.n, self.n))
    #    l = []

    #   for i in range(1, 5):
    #        for j in [True, False]:
    #            newB = np.rot90(board, i)
    #            newPi = np.rot90(pi_board, i)
    #            if j:
    #                newB = np.fliplr(newB)
    #                newPi = np.fliplr(newPi)
    #            l += [(newB, list(newPi.ravel()) + [pi[-1]])]
        l= [(board, pi)]
        return l

    def stringRepresentation(self, board):#it is not neccesary
        # 8x8 numpy array (canonical board)
        return board.tostring()

    def getScore(self, board, player):#REVIEW
        # I need to calculate the score
        b = Board(self.n)
        b.pieces = np.copy(board)
        return b.countDiff(player)

#def display(board):#it is not neccesary
#    n = board.shape[0]

#    for y in range(n):
#        print (y,"|",end="")
#    print("")
#    print(" -----------------------")
#    for y in range(n):
#        print(y, "|",end="")    # print the row #
#        for x in range(n):
#            piece = board[y][x]    # get the piece to print
#            if piece == -1: print("b ",end="")
#            elif piece == 1: print("W ",end="")
#            else:
#                if x==n:
#                    print("-",end="")
#                else:
#                    print("- ",end="")
#        print("|")

#    print("   -----------------------")
