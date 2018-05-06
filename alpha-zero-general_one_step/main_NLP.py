from Coach import Coach
from NLP.NLPGame import NLPGame as Game
#from othello.pytorch.NNet import NNetWrapper as nn
#from othello.tensorflow_2.NNet import NNetWrapper as nn
from NLP.keras_2.NNet import NNetWrapper as nn
#from othello.keras_2.NNet import NNetWrapper as nn
from utils import *

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

    if args.load_model:
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    c = Coach(g, nnet, args)
    if args.load_model:
        print("Load trainExamples from file")
        c.loadTrainExamples()
    c.learn()
