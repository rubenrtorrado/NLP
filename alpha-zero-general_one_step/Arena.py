import numpy as np
from pytorch_classification.utils import Bar, AverageMeter
import time


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

    def playGame(self, verbose=False):
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
        board = self.game.getInitBoard()
        it = 0
        while self.game.getGameEnded(board, curPlayer) == 0:
            it += 1
            if verbose:
                assert (self.display)
                print("Turn ", str(it), "Player ", str(curPlayer))
                self.display(board)
            action = players[curPlayer + 1](self.game.getCanonicalForm(board, curPlayer))

            valids = self.game.getValidMoves(self.game.getCanonicalForm(board, curPlayer), 1)

            if valids[action] == 0:
                print(action)
                assert valids[action] > 0
            board, curPlayer = self.game.getNextState(board, curPlayer, action)
        if verbose:
            assert (self.display)
            print("Game over: Turn ", str(it), "Result ", str(self.game.getGameEnded(board, 1)))
            self.display(board)
        return self.game.getGameEnded(board, 1)

    def playGames(self, num, verbose=False):
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
        finalScore1 = 0
        finalScore2 = 0

        num = int(num / 2)
        oneWon = 0
        twoWon = 0
        draws = 0
        gameResults = []
        self.player1, self.player2 = self.player1, self.player1
        for _ in range(num):
            gameResult = self.playGame(verbose=verbose)
            if gameResult == 1:
                finalScore1 += 1
            # oneWon+=1
            # elif gameResult==-1:
            #    twoWon+=1
            # else:
            #    draws+=1
            # bookkeeping + plot progress
            # gameResults.append(gameResult)
            eps += 1
            eps_time.update(time.time() - end)
            end = time.time()
            bar.suffix = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}'.format(eps=eps + 1,
                                                                                                       maxeps=maxeps,
                                                                                                       et=eps_time.avg,
                                                                                                       total=bar.elapsed_td,
                                                                                                       eta=bar.eta_td)
            bar.next()

        self.player1, self.player2 = self.player2, self.player2
        gameResults2 = []
        for _ in range(num):
            gameResult2 = self.playGame(verbose=verbose)
            if gameResult2 == 1:
                finalScore2 += 1
            # oneWon+=1
            # elif gameResult==1:
            #    twoWon+=1
            # else:
            #    draws+=1
            # bookkeeping + plot progress

            # gameResults2.append(gameResult2)
            eps += 1
            eps_time.update(time.time() - end)
            end = time.time()
            bar.suffix = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}'.format(eps=eps + 1,
                                                                                                       maxeps=num,
                                                                                                       et=eps_time.avg,
                                                                                                       total=bar.elapsed_td,
                                                                                                       eta=bar.eta_td)
            bar.next()

        bar.finish()
        # finalScore1=np.sum(gameResults)
        # finalScore2 = np.sum(gameResults2) / float(len(gameResults2))
        return finalScore1, finalScore2  # oneWon, twoWon, draws

    def playGame_demo(self, target_sentence, target_y, verbose=False):
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
        # board = self.game.getInitBoard()
        board = self.game.getInitBoard_play(target_sentence, target_y)

        it = 0
        while self.game.getGameEnded(board, curPlayer) == 0:
            it += 1
            if verbose:
                assert (self.display)
                print("Turn ", str(it), "Player ", str(curPlayer))
                self.display(board)
            action = players[curPlayer + 1](self.game.getCanonicalForm(board, curPlayer))

            valids = self.game.getValidMoves(self.game.getCanonicalForm(board, curPlayer), 1)

            if valids[action] == 0:
                print(action)
                assert valids[action] > 0
            board, curPlayer = self.game.getNextState(board, curPlayer, action)
        if verbose:
            assert (self.display)
            print("Game over: Turn ", str(it), "Result ", str(self.game.getGameEnded(board, 1)))
            self.display(board)
        return self.game.getGameEnded(board, 1), action, board

    def playGames_demo(self, num, target_sentence, target_y, verbose=False):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.

        Returns:
            oneWon: games won by player1
            twoWon: games won by player2
            draws:  games won by nobody
        """
        self.player1, self.player2 = self.player1, self.player1
        gameResult, action, board = self.playGame_demo(target_sentence, target_y, verbose=verbose)
        #   if gameResult == 1:
        #       finalScore1 += 1
        #   eps += 1
        #   eps_time.update(time.time() - end)
        #   end = time.time()
        #   bar.suffix = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}'.format(eps=eps + 1,
        #
        #                                                                                          maxeps=maxeps,
        #                                                                                                       et=eps_time.avg,
        #                                                                                                       total=bar.elapsed_td,                                                                                                      eta=bar.eta_td)

        # target_y = Y[n_sonete + eps]
        # target_sentence=X[n_sonete+eps]
        # actions.append(action)
        # bar.next()

        # full_character = [n_to_char[value] for value in actions]
        # txt = ""
        # for char in full_character:
        #    txt = txt + char
        # print(full_character)
        # print(txt)

        # bar.finish()
        # finalScore1=np.sum(gameResults)
        # finalScore2 = np.sum(gameResults2) / float(len(gameResults2))
        return action, board  # oneWon, twoWon, draws
