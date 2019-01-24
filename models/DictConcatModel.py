# -*- coding: utf-8 -*-
import copy
import json
import six
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import layers
from tensorflow.contrib import crf

import model_utils
import optimization
import utils


class DictConcatConfig(object):
    """Configuration for `DictConcatConfig`."""

    def __init__(self,
                 vocab_size,
                 embedding_size=100,
                 hidden_size=128,
                 dict_hidden_size=160,
                 num_hidden_layers=1,
                 bi_direction=True,
                 rnn_cell="lstm",
                 l2_reg_lamda=0.0001,
                 embedding_dropout_prob=0.2,
                 hidden_dropout_prob=0.2,
                 num_classes=4, ):
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

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BaselineConfig` from a Python dictionary of parameters."""
        config = DictConcatConfig(vocab_size=None)
        for (key, value) in six.iteritems(json_object):
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with tf.gfile.GFile(json_file, "r") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class DictConcatModel(object):
    '''
    Model 1
    Concating outputs of two parallel Bi-LSTMs which take feature vectors and embedding vectors
    as inputs respectively.
    '''

    def __init__(self, config: DictConcatConfig, is_training,
                 input_ids, label_ids, input_dicts, seq_length,
                 init_embedding=None):

        self.input_ids = input_ids
        self.label_ids = label_ids
        self.dict = input_dicts
        self.seq_length = seq_length
        self.is_training = is_training
        input_shape = model_utils.get_shape_list(input_ids, expected_rank=3)
        self.batch_size = input_shape[0]
        self.max_length = input_shape[1]
        self.window_size = input_shape[2]

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
            feat_size = self.window_size
            x = tf.reshape(x, [self.batch_size, -1, feat_size * config.embedding_size])

        x = model_utils.dropout(x, config.embedding_dropout_prob)

        def lstm_cell(dim):
            cell = tf.nn.rnn_cell.LSTMCell(dim, name='basic_lstm_cell')
            cell = rnn.DropoutWrapper(cell, output_keep_prob=1.0 - config.hidden_dropout_prob)
            return cell

        with tf.variable_scope('character'):
            (forward_output, backword_output), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=lstm_cell(config.hidden_size),
                cell_bw=lstm_cell(config.hidden_size),
                inputs=x,
                sequence_length=self.seq_length,
                dtype=tf.float32
            )
            output = tf.concat([forward_output, backword_output], axis=2)

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

    def get_all_results(self):
        return self.loss, -self.log_likelihood, self.label_ids, self.prediction, self.seq_length

    def get_loss(self):
        assert self.is_training, "loss can only get while training"
        return self.loss

    def get_NLLLoss(self):
        assert self.is_training
        return -self.log_likelihood, "NLLLoss can only get while training"


def model_fn_builder(config, init_checkpoint, tokenizer, learning_rate,
                     num_train_steps, num_warmup_steps, init_embedding=None):
    """Returns `model_fn` closure for TPUEstimator."""

    embedding = None
    if init_embedding is not None:
        embedding = utils.get_embedding(init_embedding, tokenizer.vocab, config.embedding_size)

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_dicts = features["input_dicts"]
        seq_length = features["seq_length"]
        label_ids = features["label_ids"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        model = DictConcatModel(
            config, is_training, input_ids, label_ids, input_dicts, seq_length, embedding)

        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        if init_checkpoint:
            (assignment_map,
             initialized_variable_names) = model_utils.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            utils.variable_summaries(var)
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        if mode == tf.estimator.ModeKeys.TRAIN:
            (total_loss, per_example_loss, label_ids, prediction, seq_length) = model.get_all_results()

            weight = tf.sequence_mask(seq_length, dtype=tf.int64)
            accuracy = tf.metrics.accuracy(label_ids, prediction, weights=weight)

            tf.summary.scalar('accuracy', accuracy[1])

            l2_reg_lamda = config.l2_reg_lamda
            clip = 5

            with tf.variable_scope('train_op'):
                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
                l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tvars if v.get_shape().ndims > 1])
                total_loss = total_loss + l2_reg_lamda * l2_loss
                grads, _ = tf.clip_by_global_norm(tf.gradients(total_loss, tvars), clip)
                global_step = tf.train.get_or_create_global_step()
                train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)

            logging_hook = tf.train.LoggingTensorHook({"accuracy": accuracy[1]}, every_n_iter=100)

            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                training_hooks=[logging_hook])
        elif mode == tf.estimator.ModeKeys.EVAL:
            (total_loss, per_example_loss, label_ids, prediction, seq_length) = model.get_all_results()
            loss = tf.metrics.mean(per_example_loss)

            weight = tf.sequence_mask(seq_length, dtype=tf.int64)
            accuracy = tf.metrics.accuracy(label_ids, prediction, weights=weight)
            metrics = {
                "eval_loss": loss,
                "eval_accuracy": accuracy
            }

            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metric_ops=metrics)
        else:
            (_, _, _, prediction, seq_length) = model.get_all_results()
            predictions = {"input_ids": input_ids,
                           "prediction": prediction,
                           "ground_truths": label_ids,
                           "length": seq_length}
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions)
        return output_spec

    return model_fn

