import codecs
import numpy as np
import gensim
import os
import tensorflow as tf
import six

import tokenization


def evaluate_word_PRF(y_pred, y):
    import itertools
    y_pred = list(itertools.chain.from_iterable(y_pred))
    y = list(itertools.chain.from_iterable(y))
    assert len(y_pred) == len(y)
    cor_num = 0
    yp_word_num = y_pred.count(2) + y_pred.count(3)
    yt_word_num = y.count(2) + y.count(3)
    start = 0
    for i in range(len(y)):
        if y[i] == 2 or y[i] == 3:
            flag = True
            for j in range(start, i + 1):
                if y[j] != y_pred[j]:
                    flag = False
            if flag:
                cor_num += 1
            start = i + 1

    P = cor_num / float(yp_word_num)
    R = cor_num / float(yt_word_num)
    F = 2 * P * R / (P + R)
    return P, R, F


def convert_word_segmentation(x, y, output_dir, output_file='result.txt'):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_file = os.path.join(output_dir, output_file)
    f = codecs.open(output_file, 'w', encoding='utf-8')
    for i in range(len(x)):
        sentence = []
        for j in range(len(x[i])):
            if y[i][j] == 2 or y[i][j] == 3:
                sentence.append(x[i][j])
                sentence.append("  ")
            else:
                sentence.append(x[i][j])
        f.write(''.join(sentence).strip() + '\n')
    f.close()


# get pre-trained embeddings
def get_embedding(embedding_file, vocab, size=100):
    init_embedding = np.zeros(shape=[len(vocab), size])
    pre_trained = gensim.models.KeyedVectors.load(embedding_file)
    pre_trained_vocab = set([w for w in pre_trained.wv.vocab.keys()])
    c = 0
    for word in vocab.keys():
        if len(word) == 1:
            if word in pre_trained_vocab:
                init_embedding[vocab[word]] = pre_trained[word]
            else:
                init_embedding[vocab[word]] = np.random.uniform(-0.5, 0.5, size)
                c += 1

    for word in vocab.keys():
        if len(word) == 2:
            first_char = word[0] if word[0] in vocab else "U"
            second_char = word[1] if word[1] in vocab else "U"
            init_embedding[vocab[word]] = (init_embedding[vocab[first_char]] + init_embedding[vocab[second_char]]) / 2

    init_embedding[vocab["P"]] = np.zeros(shape=size)
    print('oov character rate %f' % (float(c) / len(vocab)))
    return init_embedding


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def printable_text(text):
    """Returns text encoded in a way suitable for print or `tf.logging`."""

    # These functions want `str` for both Python2 and Python3, but in one case
    # it's a Unicode string and in the other it's a byte string.
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text
        elif isinstance(text, unicode):
            return text.encode("utf-8")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope(var.name.split(":")[0] + '/summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)