# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import codecs
import os
import re
import collections
import json
import os
from typing import List
import numpy as np

import tokenization
import tensorflow as tf

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string('input', None,
                    "The input data. data need to be segment by '  '")
flags.DEFINE_string('output_dir', 'data',
                    "The output data dir")
flags.DEFINE_string('output', None,
                    "The input data file name, should without suffix")
flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")
flags.DEFINE_bool("log", False,
                  "show log on the console or not")

rNUM = re.compile(r'([-+])?\d+(([.·])\d+)?%?')
rENG = re.compile(r'[A-Za-z_.]+')


def strQ2B(ustring):
    """全角转半角"""
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 12288:  # 全角空格直接转换
            inside_code = 32
        elif 65281 <= inside_code <= 65374:  # 全角字符（除空格）根据关系转化
            inside_code -= 65248
        rstring += chr(inside_code)
    return rstring


# def preprocess(input, output_dir, output, tokenizer):
#     output_info = os.path.join(output_dir, output + ".info")
#     output_file = os.path.join(output_dir, output + ".tf_record")
#
#     examples = []
#     guid = 0
#
#     with codecs.open(input, 'r', 'utf-8') as fin:
#         for line in fin:
#             sent = strQ2B(line).split()
#
#             # new_sent = []
#             # for word in sent:
#             #     word = rNUM.sub('0', word)
#             #     word = rENG.sub('X', word)
#             #     new_sent.append(word)
#
#             new_sent = sent
#
#             examples.append(convert_words_to_example(guid, new_sent))
#             guid += 1
#
#     file_based_convert_examples_to_features(examples,
#                                             tokenizer,
#                                             output_file)
#
#     info = {"size": len(examples)}
#     with codecs.getwriter("utf-8")(tf.gfile.Open(output_info, "w")) as writer:
#         writer.write(json.dumps(info))

def preprocess(input, output_dir, output):
    output_filename = os.path.join(output_dir, output)
    sents = []
    with codecs.open(input, 'r', 'utf-8') as fin:
        with codecs.open(output_filename, 'w', 'utf-8') as fout:
            for line in fin:
                sent = strQ2B(line).split()
                new_sent = []
                for word in sent:
                    word = rNUM.sub('0', word)
                    word = rENG.sub('X', word)
                    new_sent.append(word)
                sents.append(new_sent)
            for sent in sents:
                fout.write('  '.join(sent))
                fout.write('\n')


def main(_):
    if FLAGS.log:
        tf.logging.set_verbosity(tf.logging.INFO)
    output_dir = FLAGS.output_dir
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    preprocess(FLAGS.input, output_dir, FLAGS.output)


if __name__ == '__main__':
    flags.mark_flag_as_required("input")
    flags.mark_flag_as_required("output")
    flags.mark_flag_as_required("vocab_file")
    tf.app.run()

