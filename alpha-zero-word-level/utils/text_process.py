# coding=utf-8
from typing import Dict, Any

import nltk


def chinese_process(filein, fileout):
    with open(filein, 'r') as infile:
        with open(fileout, 'w') as outfile:
            for line in infile:
                output = list()
                line = nltk.word_tokenize(line)[0]
                for char in line:
                    output.append(char)
                    output.append(' ')
                output.append('\n')
                output = ''.join(output)
                outfile.write(output)


def text_to_code(tokens, dictionary, seq_len):
    code_str = ""
    eof_code = len(dictionary)
    for sentence in tokens:
        index = 0
        for word in sentence:
            code_str += (str(dictionary[word]) + ' ')
            index += 1
        while index < seq_len:
            code_str += (str(eof_code) + ' ')
            index += 1
        code_str += '\n'
    return code_str


def code_to_text(codes, dictionary):
    paras = ""
    eof_code = len(dictionary)
    for line in codes:
        numbers = map(int, line)
        for number in numbers:
            if number == eof_code:
                continue
            paras += (dictionary[str(number)] + ' ')
        paras += '\n'
    return paras


def get_tokenlized(file):
    tokenlized = list()
    with open(file) as raw:
        for text in raw:
            text = nltk.word_tokenize(text.lower())
            tokenlized.append(text)
    return tokenlized


def get_word_list(tokens):
    word_set = list()
    for sentence in tokens:
        for word in sentence:
            word_set.append(word)
    return list(set(word_set))


def get_dict(word_set):
    word_index_dict = dict()
    index_word_dict = dict()
    index = 0
    for word in word_set:
        word_index_dict[word] = str(index)
        index_word_dict[str(index)] = word
        index += 1
    return word_index_dict, index_word_dict

def text_precess(train_text_loc, test_text_loc=None):
    train_tokens = get_tokenlized(train_text_loc)
    if test_text_loc is None:
        test_tokens = list()
    else:
        test_tokens = get_tokenlized(test_text_loc)

    word_set = get_word_list(train_tokens + test_tokens)
    [word_index_dict, index_word_dict] = get_dict(word_set)

    if test_text_loc is None:
        sequence_len = len(max(train_tokens, key=len))
    else:
        sequence_len = max(len(max(train_tokens, key=len)), len(max(test_tokens, key=len)))
    with open('save/eval_data.txt', 'w') as outfile:
        encoded_text = text_to_code(test_tokens, word_index_dict, sequence_len)
        outfile.write(encoded_text)
        print('Wrote file save/eval_data.txt with length {}.'.format(len(encoded_text)))

    return sequence_len, len(word_index_dict) + 1


class Vocabulary:

    def __init__(self, word_index_dict, index_word_dict, sequence_length, vocab_size):
        self.word_index_dict = word_index_dict
        self.index_word_dict = index_word_dict
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size # type: int

    def text_to_code(self, tokens):
        return text_to_code(tokens, self.word_index_dict, self.sequence_length)

    def code_to_text(self, codes):
        return code_to_text(codes, self.index_word_dict)

    def codes_file_to_real_text_file(self, codes_file, real_text_file):
        codes = get_tokenlized(codes_file)
        with open(real_text_file, 'w') as outfile:
            outfile.write(code_to_text(codes=codes, dictionary=self.index_word_dict))

    def tokens_to_codes_file(self, tokens, codes_file):
        with open(codes_file, 'w') as outfile:
            outfile.write(text_to_code(tokens, self.word_index_dict, self.sequence_length))

    @staticmethod
    def from_word_set(word_set, sequence_length, vocab_size):
        word_index_dict, index_word_dict = get_dict(word_set)  # type: (Dict[Any, str], Dict[str, Any])
        return Vocabulary(word_index_dict, index_word_dict, sequence_length, vocab_size)