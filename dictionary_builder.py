import collections

import six
import tensorflow as tf

import utils


def load_dict(dictionary_file):
    """Loads a vocabulary file into a dictionary."""
    dictionary = collections.OrderedDict()
    index = 0
    with tf.gfile.GFile(dictionary_file, "r") as reader:
        while True:
            token = utils.convert_to_unicode(reader.readline())
            if not token:
                break
            token = token.strip()
            dictionary[token] = index
            index += 1
    return dictionary


class BasicDictionaryBuilder:
    def __init__(self, dictionary_file):
        self.dictionary = load_dict(dictionary_file)
        self.inv_dictionary = {v: k for k, v in self.dictionary.items()}

    def extract(self, tokens):
        raise NotImplementedError()


class DefaultDictionaryBuilder(BasicDictionaryBuilder):
    def __init__(self, dictionary_file, min_word_len, max_word_len):
        if not max_word_len > min_word_len:
            raise ValueError("min word length should smaller than max word length")
        self.max_word_len = max_word_len
        self.min_word_len = min_word_len
        self.dim = 2 * (max_word_len - min_word_len + 1)
        if six.PY3:
            super().__init__(dictionary_file)
        else:
            super(BasicDictionaryBuilder, self).__init__(dictionary_file)

    def extract(self, tokens):
        result = []
        for i in range(len(tokens)):
            # fw
            word_tag = []
            for l in range(self.max_word_len - 1, self.min_word_len - 2, -1):
                if (i - l) < 0:
                    word_tag.append(0)
                    continue
                word = ''.join(tokens[i - l:i + 1])
                if word in self.dictionary:
                    word_tag.append(1)
                else:
                    word_tag.append(0)
            # bw
            for l in range(self.min_word_len - 1, self.max_word_len):
                if (i + l) >= len(tokens):
                    word_tag.append(0)
                    continue
                word = ''.join(tokens[i:i + l + 1])
                if word in self.dictionary:
                    word_tag.append(1)
                else:
                    word_tag.append(0)
            result.append(word_tag)
        return result
