import model_utils
import utils
import tensorflow as tf


class SegmentModel(object):

    def __init__(self):
        self.is_training = True
        self.seq_length = 0
        self.prediction = None
        self.label_ids = None
        self.loss = None
        self.log_likelihood = None

    def get_all_results(self):
        return self.loss, -self.log_likelihood, self.label_ids, self.prediction, self.seq_length

    def get_loss(self):
        assert self.is_training, "loss can only get while training"
        return self.loss

    def get_NLLLoss(self):
        assert self.is_training
        return -self.log_likelihood, "NLLLoss can only get while training"


def model_fn_builder(segmentModel, config, init_checkpoint, tokenizer, learning_rate,
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

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        model = segmentModel(config, is_training, features, embedding)

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
            input_ids = features["input_ids"]
            label_ids = features["label_ids"]
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
