import tensorflow as tf
import BertModel
import optimization
from tensorflow.contrib import crf
from tensorflow.contrib import layers


def create_model(bert_config: BertModel.BertConfig, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings):
    """Creates a classification model."""
    model = BertModel.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    # In the demo, we are doing a simple classification task on the entire
    # segment.
    #
    # If you want to use the token-level scores, use model.get_sequence_output()
    # instead.

    # [batch_size, max_seq_length, hidden_size]
    output_layer = model.get_sequence_output()

    # remove the scores of [CLS] in the beginning
    output_layer = tf.slice(output_layer, [0, 1, 0], [-1, -1, -1])
    # [batch_size, max_seq_length]
    labels = tf.slice(labels, [0, 1], [-1, -1])
    seq_length = tf.reduce_sum(input_mask, 1)

    # subtract the length of [CLS] and [SEP]
    seq_length = tf.subtract(seq_length, 2)

    with tf.variable_scope('loss'):
        if is_training:
            # I.e., 0.1 dropout
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

        # [batch_size, seq_length, hidden_size] -> # [batch_size, seq_length, num_labels]
        scores = layers.fully_connected(inputs=output_layer, num_outputs=num_labels, activation_fn=None)

        # crf
        log_likelihood, transition_param = crf.crf_log_likelihood(
            scores, labels, seq_length)
        loss = tf.reduce_mean(-log_likelihood)
        prediction, _ = crf.crf_decode(scores, transition_param, seq_length)

        return (loss, -log_likelihood, labels, prediction, seq_length)


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_dicts"]
        segment_ids = features["seq_length"]
        label_ids = features["label_ids"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (total_loss, per_example_loss, label_ids, prediction, seq_length) = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
            num_labels, use_one_hot_embeddings)

        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = BertModel.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            if use_tpu:

                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:

            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

            logging_hook = tf.train.LoggingTensorHook({"loss": total_loss}, every_n_iter=100)

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                training_hooks=[logging_hook],
                scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:
            def metric_fn(per_example_loss, label_ids, seq_length):
                loss = tf.metrics.mean(per_example_loss)
                return {
                    "eval_loss": loss
                    # "predictions": viterbi_sequence,
                    # "ground_truths": label_ids,
                    # "length": seq_length
                }

            eval_metrics = (metric_fn, [per_example_loss, label_ids, seq_length])
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn)
        else:
            predictions = {"input_ids": input_ids,
             "prediction": prediction,
             "ground_truths": label_ids,
             "length": seq_length}
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                predictions=predictions,
                scaffold_fn=scaffold_fn)
        return output_spec

    return model_fn
