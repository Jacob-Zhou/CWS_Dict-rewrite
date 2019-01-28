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

        input_dicts = tf.sparse.to_dense(example["input_dicts"])
        input_dicts = tf.reshape(input_dicts, shape=[-1, dict_dim])
        flip_mask = tf.random.uniform(tf.shape(input_dicts)) < 0.05
        # flip if flip mask is true
        input_dicts = tf.cast(input_dicts, dtype=tf.bool)
        input_dicts = tf.logical_xor(input_dicts, flip_mask)
        input_dicts = tf.cast(input_dicts, dtype=tf.int64)
        example["input_dicts"] = input_dicts

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
            d = d.shuffle(buffer_size=50000)

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
    tf.logging.set_verbosity(tf.logging.INFO)

    tokenizer = tokenization.WindowBigramTokenizer(
        vocab_file="data/cityu/vocab.txt", bigram_file="data/empty",
        do_lower_case=False, window_size=5)

    processor = getattr(process, "CWSProcessor")()
<<<<<<< HEAD
    train_file = os.path.join("debut", "train.tf_record")
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
    train_examples = processor.get_test_examples("data/cityu")
=======
    train_examples = processor.get_train_examples(FLAGS.data_dir)
>>>>>>> parent of a75e8d3... fix bug in dict hyper model
=======
    train_examples = processor.get_train_examples(FLAGS.data_dir)
>>>>>>> parent of a75e8d3... fix bug in dict hyper model
=======
    train_examples = processor.get_train_examples(FLAGS.data_dir)
>>>>>>> parent of a75e8d3... fix bug in dict hyper model
=======
    train_file = os.path.join("debug", "train.tf_record")
    train_examples = processor.get_train_examples("data/cityu")
    dict_builder = dictionary_builder.DefaultDictionaryBuilder("data/dict/dict_2",
                                                min_word_len=2,
                                                max_word_len=5)
>>>>>>> parent of 6549cd3... broke commit
    process.file_based_convert_examples_to_features(
        examples=train_examples, tokenizer=tokenizer, dict_builder=dict_builder,
        label_map=processor.get_labels(), output_file=train_file)
    train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            batch_size=2,
            is_training=True,
            drop_remainder=True,
            input_dim=tokenizer.dim,
            dict_dim=dict_builder.dim)

    iterator = train_input_fn().make_one_shot_iterator() 
    one_element = iterator.get_next()
    with tf.Session() as sess:
        for i in range(5):
            print(sess.run(one_element))


if __name__ == "__main__":
    tf.app.run()
