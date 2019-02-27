# -*- coding: utf-8 -*-
import copy
import json
import six
import tensorflow as tf

from models.SegmentModel import SegmentModel
from .ModelConfig import ModelConfig
from tensorflow.contrib import rnn
from tensorflow.contrib import layers
from tensorflow.contrib import crf

import model_utils
import optimization
import utils


class AttendedInputConfig(ModelConfig):
    """Configuration for `DictConcatConfig`."""

    def __init__(self, vocab_size=8004, embedding_size=100, hidden_size=128, dict_hidden_size=160, num_hidden_layers=1,
                 bi_direction=True, rnn_cell="lstm", l2_reg_lamda=0.0001, embedding_dropout_prob=0.2,
                 hidden_dropout_prob=0.2, num_classes=4, **kw):
        """Constructs BertConfig.

        Args:
          vocab_size: Vocabulary size of `inputs_ids` in `BertModel`.
          hidden_size: Size of the encoder layers and the pooler layer.
          num_hidden_layers: Number of hidden layers in the Transformer encoder.
          bi_direction: The size of the "intermediate" (i.e., feed-forward)
            layer in the Transformer encoder.
          rnn_cell: The non-linear activation function (function or string) in the
            encoder and pooler.
          hidden_dropout_prob: The dropout probability for all fully connected
            layers in the embeddings, encoder, and pooler.
          embedding_dropout_prob: The dropout ratio for the attention
            probabilities.
          embedding_size: The maximum sequence length that this model might
            ever be used with. Typically set this to something large just in case
            (e.g., 512 or 1024 or 2048).
          num_classes: The vocabulary size of the `token_type_ids` passed into
            `BertModel`.
          init_embedding: The stdev of the truncated_normal_initializer for
            initializing all weight matrices.
        """
        super(AttendedInputConfig).__init__(**kw)
        self.l2_reg_lamda = l2_reg_lamda
        self.dict_hidden_size = dict_hidden_size
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.rnn_cell = rnn_cell
        self.bi_direction = bi_direction
        self.hidden_dropout_prob = hidden_dropout_prob
        self.embedding_dropout_prob = embedding_dropout_prob
        self.embedding_size = embedding_size
        self.num_classes = num_classes


class AttendedInputModel(SegmentModel):
    '''
    Model 1
    Concating outputs of two parallel Bi-LSTMs which take feature vectors and embedding vectors
    as inputs respectively.
    '''

    def __init__(self, config: AttendedInputConfig, is_training, features, init_embedding=None):

        super(AttendedInputModel).__init__()
        input_ids = features["input_ids"]
        input_dicts = features["input_dicts"]
        seq_length = features["seq_length"]
        label_ids = features["label_ids"]

        self.input_ids = input_ids
        self.label_ids = label_ids
        self.dict = input_dicts
        self.seq_length = seq_length
        self.is_training = is_training
        input_shape = model_utils.get_shape_list(input_ids, expected_rank=3)
        self.batch_size = input_shape[0]
        self.max_length = input_shape[1]
        self.window_size = input_shape[2]
        dict_shape = model_utils.get_shape_list(input_dicts, expected_rank=3)
        self.dict_dim = dict_shape[2]

        if not is_training:
            config.embedding_dropout_prob = 0.0
            config.hidden_dropout_prob = 0.0

        if init_embedding is None:
            self.embedding = tf.get_variable(shape=[config.vocab_size, config.embedding_size],
                                             dtype=tf.float32,
                                             name='embedding',
                                             initializer=tf.truncated_normal_initializer(stddev=0.02))
        else:
            self.embedding = tf.Variable(init_embedding, dtype=tf.float32, name='embedding')

        with tf.variable_scope('embedding'):
            x = tf.nn.embedding_lookup(self.embedding, self.input_ids)

        def lstm_cell(dim):
            cell = tf.nn.rnn_cell.LSTMCell(dim, name='basic_lstm_cell')
            cell = rnn.DropoutWrapper(cell, output_keep_prob=1.0 - config.hidden_dropout_prob)
            return cell

        with tf.variable_scope('dict'):
            self.dict = tf.cast(self.dict, dtype=tf.float32)
            (forward_output, backword_output), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=lstm_cell(config.dict_hidden_size),
                cell_bw=lstm_cell(config.dict_hidden_size),
                inputs=self.dict,
                sequence_length=self.seq_length,
                dtype=tf.float32
            )
            dict_output = tf.concat([forward_output, backword_output], axis=2)

        with tf.variable_scope('input_attention'):
            feat_size = self.window_size
            input_attention = layers.fully_connected(
                inputs=dict_output,
                num_outputs=feat_size,
                activation_fn=tf.sigmoid
            )
            # [B, L, F] * [B, L, F, E] -> [B, L, F, E]
            input_attention = tf.expand_dims(input_attention, -1)
            attend_input = tf.multiply(x, input_attention)
            attend_input = tf.reshape(attend_input, [self.batch_size, -1, feat_size * config.embedding_size])
            attend_input = model_utils.dropout(attend_input, config.embedding_dropout_prob)

        with tf.variable_scope('character'):
            (forward_output, backword_output), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=lstm_cell(config.hidden_size),
                cell_bw=lstm_cell(config.hidden_size),
                inputs=attend_input,
                sequence_length=self.seq_length,
                dtype=tf.float32
            )
            output = tf.concat([forward_output, backword_output], axis=2)

        with tf.variable_scope('output'):
            output = tf.concat([dict_output, output], axis=2)
            scores = layers.fully_connected(
                inputs=output,
                num_outputs=config.num_classes,
                activation_fn=None
            )
            transition_param = tf.get_variable("transitions", [config.num_classes, config.num_classes])
            self.prediction, _ = crf.crf_decode(scores, transition_param, self.seq_length)

        with tf.variable_scope('loss'):
            # crf
            self.log_likelihood, _ = crf.crf_log_likelihood(
                scores, self.label_ids, self.seq_length, transition_param)
            self.loss = tf.reduce_mean(-self.log_likelihood)
