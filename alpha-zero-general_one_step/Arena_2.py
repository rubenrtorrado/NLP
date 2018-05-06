
import numpy as np
from pytorch_classification.utils import Bar, AverageMeter
import time
from MCTS import MCTS

class Arena():
    """
    An Arena class where any 2 agents can be pit against each other.
    """
    def __init__(self, player1, player2, game, display=None):
        """
        Input:
            player 1,2: two functions that takes board as input, return action
            game: Game object
            display: a function that takes board as input and prints it (e.g.
                     display in othello/OthelloGame). Is necessary for verbose
                     mode.

        see othello/OthelloPlayers.py for an example. See pit.py for pitting
        human players/other baselines with each other.
        """
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.display = display

    def playGame(self,sonete,sequences,nround, verbose=False):
        """
        Executes one episode of a game.

        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """
        players = [self.player1, self.player1, self.player1]
        curPlayer = 1
        board = self.game.getInitBoard_fix_sonet(sonete,sequences,nround)
        it = 0
        while self.game.getGameEnded(board, curPlayer)==0:
            it+=1
            if verbose:
                assert(self.display)
                print("Turn ", str(it), "Player ", str(curPlayer))
                self.display(board)
            action = players[curPlayer+1](self.game.getCanonicalForm(board, curPlayer))

            valids = self.game.getValidMoves(self.game.getCanonicalForm(board, curPlayer),1)

            if valids[action]==0:
                print(action)
                assert valids[action] >0
            board, curPlayer = self.game.getNextState(board, curPlayer, action)
        if verbose:
            assert(self.display)
            print("Game over: Turn ", str(it), "Result ", str(self.game.getGameEnded(board, 1)))
            self.display(board)
        return action,board

    def playGames(self, num, sonete,secuence,mcts,verbose=False):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.

        Returns:
            oneWon: games won by player1
            twoWon: games won by player2
            draws:  games won by nobody
        """
        eps_time = AverageMeter()
        bar = Bar('Arena.playGames', max=num)
        end = time.time()
        eps = 0
        maxeps = int(num)
        finalScore1=0
        finalScore2=0

        num = int(num/2)
        oneWon = 0
        twoWon = 0
        draws = 0
        gameResults=[]
        global nround
        actions=[]
        self.player1, self.player2 = self.player1, self.player1
        board = self.game.getInitBoard()
        for i in range(100):
            nround = i
            #action,sonete = self.playGame(sonete,sequences,nround,verbose=verbose)
            pi = mcts.getActionProb(sonete, temp=1)
            #actions.append(action)

            eps_time.update(time.time() - end)
            end = time.time()


        return actions#finalScore1, finalScore2#oneWon, twoWon, draws
