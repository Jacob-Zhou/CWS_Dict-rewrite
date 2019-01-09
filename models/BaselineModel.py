# -*- coding: utf-8 -*-
import copy
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import layers
from tensorflow.contrib import crf
import six
import json
import model_utils
import utils
import optimization

__all__ = ["BaselineConfig", "BaselineModel", "model_fn_builder"]


class BaselineConfig(object):
    """Configuration for `BaselineModel`."""

    def __init__(self,
                 vocab_size,
                 embedding_size=100,
                 hidden_size=128,
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
        config = BaselineConfig(vocab_size=None)
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


class BaselineModel(object):
    '''
    Baseline models
    BiLSTM+CRF and Stacked BiLSTM+CRF
    '''
    def __init__(self, config: BaselineConfig, is_training, input_ids, label_ids, seq_length, init_embedding=None):
        """Constructor for BertModel.

        Args:
          config: `BertConfig` instance.
          is_training: bool. rue for training model, false for eval model. Controls
            whether dropout will be applied.
          input_ids: int64 Tensor of shape [batch_size, seq_length, feat_size].
          label_ids: (optional) int64 Tensor of shape [batch_size, seq_length].
          seq_length: (optional) int64 Tensor of shape [batch_size].
          init_embedding: (optional)

        Raises:
          ValueError: The config is invalid or one of the input tensor shapes
            is invalid.
        """
        self.input_ids = input_ids
        self.label_ids = label_ids
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

        with tf.variable_scope('rnn_cell'):
            if config.rnn_cell == 'lstm':
                self.fw_cell = tf.nn.rnn_cell.LSTMCell(config.hidden_size, name='basic_lstm_cell')
                self.bw_cell = tf.nn.rnn_cell.LSTMCell(config.hidden_size, name='basic_lstm_cell')
            else:
                self.fw_cell = rnn.GRUCell(config.hidden_size)
                self.bw_cell = rnn.GRUCell(config.hidden_size)
            self.fw_cell = rnn.DropoutWrapper(self.fw_cell, output_keep_prob=1.0 - config.hidden_dropout_prob)
            self.bw_cell = rnn.DropoutWrapper(self.bw_cell, output_keep_prob=1.0 - config.hidden_dropout_prob)
            self.fw_multi_cell = rnn.MultiRNNCell([self.fw_cell] * config.num_hidden_layers)
            self.bw_multi_cell = rnn.MultiRNNCell([self.bw_cell] * config.num_hidden_layers)

        with tf.variable_scope('rnn'):
            if config.bi_direction:
                (forward_output, backword_output), _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=self.fw_multi_cell,
                    cell_bw=self.bw_multi_cell,
                    inputs=x,
                    sequence_length=self.seq_length,
                    dtype=tf.float32
                )
                output = tf.concat([forward_output, backword_output], axis=2)
            else:
                print('bi_direction is false')
                forward_output, _ = tf.nn.dynamic_rnn(
                    cell=self.fw_multi_cell,
                    inputs=x,
                    sequence_length=self.seq_length,
                    dtype=tf.float32
                )
                output = forward_output

        with tf.variable_scope('output'):
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
        seq_length = features["seq_length"]
        label_ids = features["label_ids"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        model = BaselineModel(
            config, is_training, input_ids, label_ids, seq_length, embedding)

        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        if init_checkpoint:
            (assignment_map,
             initialized_variable_names) = model_utils.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        if mode == tf.estimator.ModeKeys.TRAIN:
            (total_loss, per_example_loss, label_ids, prediction, seq_length) = model.get_all_results()
            loss = tf.metrics.mean(per_example_loss)

            weight = tf.sequence_mask(seq_length, dtype=tf.int64)
            accuracy = tf.metrics.accuracy(label_ids, prediction, weights=weight)

            tf.summary.scalar('loss', loss[1])
            tf.summary.scalar('accuracy', accuracy[1])

            # train_op = optimization.create_optimizer(
            #     total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu=False)

            l2_reg_lamda = config.l2_reg_lamda
            clip = 5

            with tf.variable_scope('train_op'):
                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
                l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tvars if v.get_shape().ndims > 1])
                total_loss = total_loss + l2_reg_lamda * l2_loss
                grads, _ = tf.clip_by_global_norm(tf.gradients(total_loss, tvars), clip)
                global_step = tf.train.get_or_create_global_step()
                train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)

            logging_hook = tf.train.LoggingTensorHook({"loss": total_loss,
                                                       "accuracy": accuracy[1]}, every_n_iter=100)

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
                "accuracy": accuracy
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


