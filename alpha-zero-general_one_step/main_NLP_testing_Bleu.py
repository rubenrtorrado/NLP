from Coach_testing_Bleu import Coach
from NLP.NLPGame import NLPGame as Game
#from othello.pytorch.NNet import NNetWrapper as nn
#from othello.tensorflow_2.NNet import NNetWrapper as nn
from NLP.keras_2.NNet import NNetWrapper as nn
#from othello.keras_2.NNet import NNetWrapper as nn
from utils import *
import numpy as np
from MCTS_Bleu import MCTS

args = dotdict({
    'numIters': 100,
    'numEps': 100,
    'tempThreshold': 15,
    'updateThreshold': 0.6,
    'maxlenOfQueue': 200000,
    'numMCTSSims': 25,
    'arenaCompare': 10,
    'cpuct': 1,

    'checkpoint': './temp/',
    'load_model': True,
    #'load_folder_file': ('/dev/models/8x100x50','best.pth.tar'),
    'load_folder_file': ('./temp','checkpoint_343.pth.tar'),
    'numItersForTrainExamplesHistory': 20,

})

if __name__=="__main__":
    g = Game(100)#done
    nnet = nn(g)
    target_sonnets=99

    if args.load_model:
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    tmcts = MCTS(g, nnet, args)

    c = Coach(lambda x: np.argmax(tmcts.getActionProb(x, temp=0)),lambda x: np.argmax(tmcts.getActionProb(x, temp=0)),g, nnet, args)
    if args.load_model:
        print("Load trainExamples from file")
        c.loadTrainExamples()
    c.test(target_sonnets)
