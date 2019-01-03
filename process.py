import collections
from typing import *
import os
import tensorflow as tf
import csv
import tokenization
import numpy as np

import utils


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, labels=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text
        self.labels = labels


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_dicts, label_ids, seq_length):
        self.input_ids = input_ids
        self.input_dicts = input_dicts
        self.seq_length = seq_length
        self.label_ids = label_ids


def convert_single_example(ex_index, example: InputExample,
                           tokenizer, label_map, dict_builder=None):
    """Converts a single `InputExample` into a single `InputFeatures`."""
    # label_map = {"B": 0, "M": 1, "E": 2, "S": 3}

    tokens_raw = tokenizer.tokenize(example.text)
    labels_raw = example.labels

    # Account for [CLS] and [SEP] with "- 2"

    # The convention in BERT is:
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = []
    label_ids = []
    for token, label in zip(tokens_raw, labels_raw):
        tokens.append(token)
        label_ids.append(label_map[label])

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    if dict_builder is None:
        input_dicts = np.zeros_like(tokens_raw, dtype=np.int64)
    else:
        input_dicts = dict_builder.extract(tokens)
    seq_length = len(tokens)
    assert seq_length == len(input_ids)
    assert seq_length == len(input_dicts)
    assert seq_length == len(label_ids)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.

    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid:        %s" % example.guid)
        tf.logging.info("tokens:      %s" % " ".join(
            [utils.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids:   %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_ids:   %s" % " ".join([str(x) for x in input_dicts]))
        tf.logging.info("labels:      %s" % " ".join([str(x) for x in example.labels]))
        tf.logging.info("labels_ids:  %s" % " ".join([str(x) for x in label_ids]))

    feature = InputFeatures(
        input_ids=input_ids,
        input_dicts=input_dicts,
        label_ids=label_ids,
        seq_length=seq_length)
    return feature


def file_based_convert_examples_to_features(
        examples, tokenizer, label_map, output_file, dict_builder=None):
    """Convert a set of `InputExample`s to a TFRecord file."""

    writer = tf.python_io.TFRecordWriter(output_file)

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, tokenizer, label_map, dict_builder)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        plain_input_ids = np.array(feature.input_ids).reshape([-1])
        features["input_ids"] = create_int_feature(plain_input_ids)
        plain_input_dicts = np.array(feature.input_dicts).reshape([-1])
        features["input_dicts"] = create_int_feature(plain_input_dicts)
        features["seq_length"] = create_int_feature([feature.seq_length])
        features["label_ids"] = create_int_feature(feature.label_ids)

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())


def convert_words_to_example(guid: int, sentence: List[str]) -> InputExample:
    def word2tag(word):
        if len(word) == 1:
            return ["S"]
        if len(word) == 2:
            return ["B", "E"]
        tag = ["B"]
        for i in range(1, len(word) - 1):
            tag.append("M")
        tag.append("E")
        return tag

    labels = []
    text = ' '.join(''.join(sentence))
    for word in sentence:
        labels += word2tag(word)
    return InputExample(guid, text, labels)


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the map of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_file(cls, input_file):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            return f.readlines()


class CWSProcessor(DataProcessor):
    """Processor for the XNLI data set."""

    def __init__(self):
        self.language = "zh"

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_file(os.path.join(data_dir, "train")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_file(os.path.join(data_dir, "dev")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_file(os.path.join(data_dir, "test")), "test")

    def get_labels(self):
        """See base class."""
        return {"B": 0, "M": 1, "E": 2, "S": 3}

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            # Only the test set has a header
            guid = "%s-%s" % (set_type, i)
            text = utils.convert_to_unicode(line.strip())
            labels = self._labels_words(text)
            examples.append(
                InputExample(guid=guid, text=text, labels=labels))
        return examples

    @staticmethod
    def _labels_words(text):
        def word2label(w):
            if len(w) == 1:
                return ["S"]
            if len(w) == 2:
                return ["B", "E"]
            label = ["B"]
            for i in range(1, len(w) - 1):
                label.append("M")
            label.append("E")
            return label

        words = text.split()
        labels = []
        for word in words:
            labels += word2label(word)
        return "".join(labels)
