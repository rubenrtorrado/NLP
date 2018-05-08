import time
import numpy as np

from models.mcts.pytorch_classification.utils import Bar, AverageMeter

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

        see othello/TextGenPlayers.py for an example. See pit.py for pitting
        human players/other baselines with each other.
        """
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.display = display

    # def playGame(self, verbose=False):
    #     """
    #     Executes one episode of a game.
    #
    #     Returns:
    #         either
    #             winner: player who won the game (1 if player1, -1 if player2)
    #         or
    #             draw result returned from the game that is neither 1, -1, nor 0.
    #     """
    #     # players = [self.player2, None, self.player1]
    #     board = self.game.getInitBoard()
    #
    #     # Player 1 turn
    #     scoreP1 = self.playTurn(board, self.player1, 1, verbose)
    #     # Player 2 turn
    #     scoreP2 = self.playTurn(board, self.player2, 2, verbose)
    #
    #     if verbose:
    #         assert(self.display)
    #         print("Game over: Turn ", str(it), "Result ", str(self.game.getGameEnded(board, 1)))
    #         self.display(board)
    #
    #     # return self.game.getGameEnded(board, 1)
    #     return 1 if scoreP1 > scoreP2 else if scoreP2 > scoreP1 -1 else

    def playTurn(self, board, player, playerNumber, verbose):
        it = 0
        score = self.game.getGameEnded(board, playerNumber)
        # while score == -1:  # FIXME
        while score == 0:
            it += 1
            if verbose:
                assert (self.display)
                print("Move ", str(it), "Player ", str(playerNumber))
                self.display(board)
            action = player(board)
            valids = self.game.getValidMoves(board, 2)

            if valids[action] == 0:
                print(action)
                assert valids[action] > 0
            board, _ = self.game.getNextState(board, playerNumber, action)
            score = self.game.getGameEnded(board, playerNumber)

        return score
            
    def playGames(self, num, verbose=True):
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

        # num = int(num/2)
        oneWon = 0
        twoWon = 0
        draws = 0

        for gameNumber in range(num):

            board = self.game.getInitBoard()

            # Player 1 turn
            scoreP1 = self.playTurn(board, self.player1, 1, verbose)
            # Player 2 turn
            scoreP2 = self.playTurn(board, self.player2, 2, verbose)

            if verbose:
                assert (self.display)
                print("Game over:  Game #={} P1={} P2={} Result={}".format(
                    str(gameNumber), scoreP1, scoreP2, str(int(scoreP1 > scoreP2))))
                self.display(board)

            # return self.game.getGameEnded(board, 1)
            if scoreP1 > scoreP2:
                oneWon += 1
            elif scoreP2 > scoreP1:
                twoWon += 1
            else:
                draws += 1
            # bookkeeping + plot progress
            eps += 1
            eps_time.update(time.time() - end)
            end = time.time()
            bar.suffix  = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}'.format(eps=eps+1, maxeps=maxeps, et=eps_time.avg,
                                                                                                       total=bar.elapsed_td, eta=bar.eta_td)
            bar.next()

        # self.player1, self.player2 = self.player2, self.player1
        #
        # for _ in range(num):
        #     gameResult = self.playGame(verbose=verbose)
        #     if gameResult==-1:
        #         oneWon+=1
        #     elif gameResult==1:
        #         twoWon+=1
        #     else:
        #         draws+=1
        #     # bookkeeping + plot progress
        #     eps += 1
        #     eps_time.update(time.time() - end)
        #     end = time.time()
        #     bar.suffix  = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}'.format(eps=eps+1, maxeps=num, et=eps_time.avg,
        #                                                                                                total=bar.elapsed_td, eta=bar.eta_td)
        #     bar.next()
            
        bar.finish()

        return oneWon, twoWon, draws
