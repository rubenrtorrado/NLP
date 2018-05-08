from random import randint

import numpy as np


class Board():

    def __init__(self, sequence_length, end_token, original=None, mask=None, tokens=None, mask_size=None):
        "Set up initial board configuration."
        self.sequence_length = sequence_length
        self.end_token = end_token
        self.original = list(original) if original != None else []
        self.mask_size = mask_size
        self.mask = self.__create_mask(original) if mask == None else list(mask)
        self.tokens = [] if tokens == None else list(tokens)

    def __create_mask(self, original):
        """
        Creates a mask of random size in a random position.
        """
        mask_size = min(randint(1, 8), len(original) / 2)  # mask size is half of the sentence
        mask_position = min(randint(0, 5), len(original) - mask_size)  # anywhere that the mask fits
        mask = [token for token in original]
        # print('mask_size', mask_size, 'mask_position', mask_position)
        for i in range(mask_position, mask_position + mask_size):
            mask[i] = '_'

        self.mask_size = mask_size
        print('\nGenerated mask:\n    m(x)=[{}]\noriginal=[{}]\n'.format(
            ','.join([str(i) for i in mask]),
            ','.join([str(i) for i in original])))

        return mask

    def getModelInput(self):
        input = list(self.tokens)
        while len(input) < self.sequence_length:
            input.append(self.end_token)
        return np.array(input)


    def correct_sentence(self):
        """
        Counts the # of generated words that match the original sentence
        """
        for i in range(len(self.tokens)):
            if self.tokens[i] != self.original[i]:
                return -1

        return 1
    def correct_tokens(self):
        """
        Counts the # of generated words that match the original sentence
        """
        count = 0
        for i in range(len(self.tokens)):
            if self.mask[i] == '_' and self.tokens[i] == self.original[i]:
                count += 1

        return count

    def to_string(self):
        original = ','.join([str(i) for i in self.original])
        tokens = ','.join([str(i) for i in self.tokens])
        return '[ tokens={}\noriginal={}]'.format(tokens, original)

    @staticmethod
    def copy(board):
        b = Board(board.sequence_length, board.end_token,
                  board.original, board.mask, board.tokens, board.mask_size)
        return b