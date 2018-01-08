import os
import pdb
import sys
import traceback
import numpy as np
import tensorflow as tf
from IPython.terminal.embed import embed
sys.path.insert(0, os.path.abspath("."))
from capsule import _renorm_r


def test1():
    """Convolved over image [1, 7, 7, 1] with out_channels=1,
    kernel_size=3x3, stride=1

    Number of upper units convolved lower units:

    1 2 3 3 3 2 1
    2 4 6 6 6 4 2
    3 6 9 9 9 6 3
    3 6 9 9 9 6 3
    3 6 9 9 9 6 3
    2 4 6 6 6 4 2
    1 2 3 3 3 2 1

    So res[0, 0, 0] should be:

    1/1 1/2 1/3
    1/2 1/4 1/7
    1/3 1/7 1/9

    and res[0, 2, 2] should be

    1/9 1/9 1/9
    1/9 1/9 1/9
    1/9 1/9 1/9
    """
    img_shape = [5, 5, 1, 9, 1]
    inputs = tf.placeholder(tf.float32, [None] + img_shape)
    r = _renorm_r(inputs, 1)

    config = tf.ConfigProto(device_count={'GPU': 0})
    with tf.Session(config=config) as sess:
        res = sess.run(r,
                       feed_dict={inputs: np.ones([1] + img_shape)})

    print(res)

    # ipython interactive terminal
    embed()


def test2():
    """Convolved over image [1, 7, 7, 2] with out_channels=2,
    kernel_size=3x3, stride=1

    Number of upper units convolved lower units:

    1 2 3 3 3 2 1
    2 4 6 6 6 4 2
    3 6 9 9 9 6 3
    3 6 9 9 9 6 3
    3 6 9 9 9 6 3
    2 4 6 6 6 4 2
    1 2 3 3 3 2 1

    So res[0, 0, 0] should be:

    1/2 1/4  1/6
    1/4 1/8  1/14
    1/6 1/14 1/18
    1/2 1/4  1/6
    1/4 1/8  1/14
    1/6 1/14 1/18

    and res[0, 2, 2] should be

    1/18 1/18 1/18
    1/18 1/18 1/18
    1/18 1/18 1/18
    1/18 1/18 1/18
    1/18 1/18 1/18
    1/18 1/18 1/18
    """
    img_shape = [5, 5, 2, 9, 2]
    inputs = tf.placeholder(tf.float32, [None] + img_shape)
    r = _renorm_r(inputs, 1)

    config = tf.ConfigProto(device_count={'GPU': 0})
    with tf.Session(config=config) as sess:
        res = sess.run(r,
                       feed_dict={inputs: np.ones([1] + img_shape)})

    print(res)

    # ipython interactive terminal
    embed()


def test3():
    """Convolved over image [1, 7, 7, 2] with out_channels=2,
    kernel_size=3x3, stride=2

    Number of upper units convolved lower units:

    1 1 2 1 2 1 1
    1 1 2 1 2 1 1
    2 2 4 2 4 2 2
    1 1 2 1 2 1 1
    2 2 4 2 4 2 2
    1 1 2 1 2 1 1
    1 1 2 1 2 1 1

    So res[0, 0, 0] should be:

    1/1 1/1 1/2
    1/1 1/1 1/2
    1/2 1/2 1/4

    and res[0, 1, 1] should be

    1/4 1/2 1/4
    1/2 1/1 1/2
    1/4 1/2 1/4
    """
    img_shape = [3, 3, 1, 9, 1]
    inputs = tf.placeholder(tf.float32, [None] + img_shape)
    r = _renorm_r(inputs, 2)

    config = tf.ConfigProto(device_count={'GPU': 0})
    with tf.Session(config=config) as sess:
        res = sess.run(r,
                       feed_dict={inputs: np.ones([1] + img_shape)})

    print(res)

    # ipython interactive terminal
    embed()


def main():
    test1()
    test2()
    test3()


if __name__ == '__main__':
    try:
        main()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
