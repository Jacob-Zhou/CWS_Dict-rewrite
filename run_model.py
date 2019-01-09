import json
import utils
import os
from models import BaselineModel
from models import DictConcatModel
from models import AttendedDictModel
from models import AttendedInputModel
from models import BiLSTMModel
import tokenization
import dictionary_builder
import tensorflow as tf
import process

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tf_record files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The scores directory where the model checkpoints will be written.")

flags.DEFINE_string(
    "model", None,
    "baseline or dict_concat")

## Other parameters
flags.DEFINE_string("bigram_file", None,
                    "The bigram file that the BERT model was trained on.")

flags.DEFINE_string("dict_file", None,
                    "The bigram file that the BERT model was trained on.")

flags.DEFINE_integer("min_word_len", 2,
                     "The min word length.")

flags.DEFINE_integer("max_word_len", 5,
                     "The max word length.")

flags.DEFINE_integer("window_size", 5,
                     "The max word length.")

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", False,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_bool(
    "early_stop", True,
    "Whether to use early stop.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_string("init_embedding", None, "Initial Embedding.")

flags.DEFINE_string("processor", "CWSProcessor", "BiLabelProcessor or CWSProcessor")


def file_based_input_fn_builder(input_file, batch_size, is_training,
                                drop_remainder, input_dim=5, dict_dim=1):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "input_ids": tf.VarLenFeature(tf.int64),
        "input_dicts": tf.VarLenFeature(tf.int64),
        "label_ids": tf.VarLenFeature(tf.int64),
        "seq_length": tf.FixedLenFeature([], tf.int64)
    }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        example["input_ids"] = tf.sparse.to_dense(example["input_ids"])
        example["input_ids"] = tf.reshape(example["input_ids"], shape=[-1, input_dim])

        example["input_dicts"] = tf.sparse.to_dense(example["input_dicts"])
        example["input_dicts"] = tf.reshape(example["input_dicts"], shape=[-1, dict_dim])

        example["label_ids"] = tf.sparse.to_dense(example["label_ids"])
        example["label_ids"] = tf.reshape(example["label_ids"], shape=[-1])
        example["seq_length"] = example["seq_length"]

        return example

    def input_fn():
        """The actual input function."""

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=1000)

        d = d.map(map_func=lambda record: _decode_record(record, name_to_features))
        d = d.padded_batch(batch_size=batch_size,
                           padded_shapes={"input_ids": [None, input_dim],
                                          "input_dicts": [None, dict_dim],
                                          "label_ids": [None],
                                          "seq_length": []},
                           drop_remainder=drop_remainder)

        return d

    return input_fn


def get_info(info_file):
    info = json.load(open(info_file, "r"))
    assert isinstance(info, dict), "wrong type of json file"
    return info


def main(_):
    if FLAGS.do_train:
        tf.logging.set_verbosity(tf.logging.INFO)

    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
        raise ValueError(
            "At least one of `train`, `eval` or `predict' must be select.")


    tf.gfile.MakeDirs(FLAGS.output_dir)

    if FLAGS.bigram_file is not None:
        tokenizer = tokenization.WindowBigramTokenizer(
            vocab_file=FLAGS.vocab_file, bigram_file=FLAGS.bigram_file,
            do_lower_case=FLAGS.do_lower_case, window_size=FLAGS.window_size)
    else:
        tokenizer = tokenization.WindowTokenizer(
            vocab_file=FLAGS.vocab_file,
            do_lower_case=FLAGS.do_lower_case, window_size=FLAGS.window_size)
    # fix me window_size

    dict_builder = None
    if FLAGS.dict_file is not None:
        dict_builder = dictionary_builder.DefaultDictionaryBuilder(FLAGS.dict_file,
                                                                   min_word_len=FLAGS.min_word_len,
                                                                   max_word_len=FLAGS.max_word_len)

    run_config = tf.estimator.RunConfig(
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps)

    processor = getattr(process, FLAGS.processor)()

    train_examples = None
    num_early_steps = None
    num_train_steps = None
    num_warmup_steps = None
    if FLAGS.do_train:
        train_examples = processor.get_train_examples(FLAGS.data_dir)
        num_train_steps = int(
            len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        num_early_steps = int(
            len(train_examples) / FLAGS.train_batch_size * 5)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    model_fn = None
    if FLAGS.model == "baseline":
        config = BaselineModel.BaselineConfig.from_json_file(FLAGS.config_file)
        model_fn = BaselineModel.model_fn_builder(
            config=config,
            init_checkpoint=FLAGS.init_checkpoint,
            learning_rate=FLAGS.learning_rate,
            tokenizer=tokenizer,
            num_train_steps=num_train_steps,
            num_warmup_steps=num_warmup_steps,
            init_embedding=FLAGS.init_embedding)
    elif FLAGS.model == "dict_concat":
        config = DictConcatModel.DictConcatConfig.from_json_file(FLAGS.config_file)
        model_fn = DictConcatModel.model_fn_builder(
            config=config,
            init_checkpoint=FLAGS.init_checkpoint,
            learning_rate=FLAGS.learning_rate,
            tokenizer=tokenizer,
            num_train_steps=num_train_steps,
            num_warmup_steps=num_warmup_steps,
            init_embedding=FLAGS.init_embedding)
    elif FLAGS.model == "attend_dict":
        config = AttendedDictModel.AttendDictConfig.from_json_file(FLAGS.config_file)
        model_fn = AttendedDictModel.model_fn_builder(
            config=config,
            init_checkpoint=FLAGS.init_checkpoint,
            learning_rate=FLAGS.learning_rate,
            tokenizer=tokenizer,
            num_train_steps=num_train_steps,
            num_warmup_steps=num_warmup_steps,
            init_embedding=FLAGS.init_embedding)
    elif FLAGS.model == "attend_input":
        config = AttendedInputModel.AttendInputConfig.from_json_file(FLAGS.config_file)
        model_fn = AttendedInputModel.model_fn_builder(
            config=config,
            init_checkpoint=FLAGS.init_checkpoint,
            learning_rate=FLAGS.learning_rate,
            tokenizer=tokenizer,
            num_train_steps=num_train_steps,
            num_warmup_steps=num_warmup_steps,
            init_embedding=FLAGS.init_embedding)
    elif FLAGS.model == "bilstm":
        config = BiLSTMModel.BiLSTMConfig.from_json_file(FLAGS.config_file)
        model_fn = BiLSTMModel.model_fn_builder(
            config=config,
            init_checkpoint=FLAGS.init_checkpoint,
            learning_rate=FLAGS.learning_rate,
            tokenizer=tokenizer,
            num_train_steps=num_train_steps,
            num_warmup_steps=num_warmup_steps,
            init_embedding=FLAGS.init_embedding)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config)

    if FLAGS.do_train:
        train_file = os.path.join(FLAGS.data_dir, "train.tf_record")
        process.file_based_convert_examples_to_features(
            examples=train_examples, tokenizer=tokenizer, dict_builder=dict_builder,
            label_map=processor.get_labels(), output_file=train_file)
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            batch_size=FLAGS.train_batch_size,
            is_training=True,
            drop_remainder=True,
            input_dim=tokenizer.dim,
            dict_dim=dict_builder.dim if dict_builder is not None else 1)

        eval_input_fn = None
        if FLAGS.do_eval:
            dev_file = os.path.join(FLAGS.data_dir, "dev.tf_record")
            dev_examples = processor.get_dev_examples(FLAGS.data_dir)
            process.file_based_convert_examples_to_features(
                examples=dev_examples, tokenizer=tokenizer, dict_builder=dict_builder,
                label_map=processor.get_labels(), output_file=dev_file)
            tf.logging.info("***** Running evaluation *****")
            tf.logging.info("  Num examples = %d", len(dev_examples))
            tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

            eval_input_fn = file_based_input_fn_builder(
                input_file=dev_file,
                batch_size=FLAGS.eval_batch_size,
                is_training=False,
                drop_remainder=False,
                input_dim=tokenizer.dim,
                dict_dim=dict_builder.dim if dict_builder is not None else 1)

        if FLAGS.early_stop:
            print("using early stop")
            assert eval_input_fn is not None, "early_stop request do_eval"
            early_stopping = tf.contrib.estimator.stop_if_no_increase_hook(
                estimator,
                metric_name='accuracy',
                max_steps_without_increase=num_early_steps,
                min_steps=num_train_steps,
                run_every_secs=None,
                run_every_steps=1000)

            tf.estimator.train_and_evaluate(estimator,
                                            train_spec=tf.estimator.TrainSpec(train_input_fn, hooks=[early_stopping]),
                                            eval_spec=tf.estimator.EvalSpec(eval_input_fn, throttle_secs=60))
        else:
            if FLAGS.do_eval:
                print("do not use early stop")
                tf.estimator.train_and_evaluate(estimator,
                                                train_spec=tf.estimator.TrainSpec(train_input_fn,
                                                                                  max_steps=num_train_steps),
                                                eval_spec=tf.estimator.EvalSpec(eval_input_fn, throttle_secs=60))
            else:
                estimator.train(train_input_fn, max_steps=num_train_steps)

    if FLAGS.do_predict:
        test_file = os.path.join(FLAGS.data_dir, "test.tf_record")
        test_examples = processor.get_test_examples(FLAGS.data_dir)
        process.file_based_convert_examples_to_features(
            examples=test_examples, tokenizer=tokenizer, dict_builder=dict_builder,
            label_map=processor.get_labels(), output_file=test_file)
        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Num examples = %d", len(test_examples))
        tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

        predict_input_fn = file_based_input_fn_builder(
            input_file=test_file,
            batch_size=FLAGS.predict_batch_size,
            is_training=False,
            drop_remainder=False,
            input_dim=tokenizer.dim,
            dict_dim=dict_builder.dim if dict_builder is not None else 1)
        predictions = []
        ground_truths = []
        texts = []
        for result in estimator.predict(input_fn=predict_input_fn, yield_single_examples=True):
            input_ids = result["input_ids"].astype(int)
            prediction = result["prediction"].astype(int)
            ground_truth = result["ground_truths"].astype(int)
            length = int(result["length"])
            if length == 0:
                continue
            tokens = tokenizer.convert_ids_to_tokens(input_ids[:length])
            predictions.append(prediction[:length].tolist())
            ground_truths.append(ground_truth[:length].tolist())
            text = [utils.printable_text(x) for x in tokens]
            texts.append(text)
        P, R, F = processor.evaluate_word_PRF(predictions, ground_truths)
        print('%s Test: P:%f R:%f F:%f' % (FLAGS.data_dir, P, R, F))
        processor.convert_word_segmentation(texts, predictions, FLAGS.output_dir, "predict")
        processor.convert_word_segmentation(texts, ground_truths, FLAGS.output_dir, "predict_golden")


if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("config_file")
    flags.mark_flag_as_required("output_dir")
    flags.mark_flag_as_required("model")
    tf.enable_eager_execution()
    tf.app.run()
