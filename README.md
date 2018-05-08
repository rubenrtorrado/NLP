# AlphaZero NLP

AlphaZero code for text generation.

This code is an adaptation of Alpha Zero General implementation:

https://github.com/suragnair/alpha-zero-general

We develop a new game which generates a sequence of characters to generate text.

Another implementation of word-level text generation is available at `alpha-zero-word-level/models/mcts/`. To run it, make sure to install all dependencies by running the following commands:

```
cd alpha-zero-word-level/
conda env create -f environment.yml
source activate alpha-zero-word-level
python main.py -t real -g mctsgan -d data/image_coco.txt
```

## Run

cd alpha-zero-general_one_step
python main_NLP.py

## License

The gem is available as open source under the terms of the [Apache License 2.0](https://opensource.org/licenses/Apache-2.0).
