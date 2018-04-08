# GYM NLP

An [OpenAI Gym](gym.openai.com) environment for Natural Language Processing.

## Installation

- Follow the installation instructions for the OpenAI Gym from its [repository](https://github.com/openai/gym)
- Install Tensorflow
- To run the iPython notebooks you will also need to `pip install jupyter baseline`, then in the `nlp-gym` folder call `jupyter notebook` from the shell.


## TO DO LIST

In order to create the first environment of OpenAI for NLP we are working in the next steps:

1) Finish step and reset function based on LSTM encoder and decoder network to generate the tuple (state, reward and terminal state ). In our case:
  a) state (EnSen, DecSen)
  b) reward BLUE similarity metric
  c) terminal state End Of Sentence
2) Propose and implement new architecture for the Deep Network Sentence
3) Test DQN, Prioritized, Dueling DQN and A3C



## License

The gem is available as open source under the terms of the [Apache License 2.0](https://opensource.org/licenses/Apache-2.0).
