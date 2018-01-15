import pdb
import tensorflow as tf
from losses import sparse_seperate_margin_loss, sparse_spread_loss
from matrix_cnn import matrix_cnn
from tf_classifier_base import TFClassifierBase


class CNNClassifier(TFClassifierBase):
    def __init__(self, *args,
                 testing=False, loss_fn='crossentropy', **kwargs):
        self._testing = testing
        self._loss_fn = loss_fn
        super(CNNClassifier, self).__init__(*args, **kwargs)

    def _build_model(self):
        placeholders = \
            {'x': tf.placeholder(
                tf.float32,
                shape=(None,
                       self._data_shape[0],
                       self._data_shape[1],
                       self._data_shape[2]),
                name='x'),
             'y': tf.placeholder(
                 tf.int32,
                 shape=(None,),
                 name='y'),
             'training': tf.placeholder(tf.bool, name='training')}

        augmented = augmentate(placeholders['x'], placeholders['training'],
                               testing=self._testing)

        # conv0
        conv0 = tf.layers.conv2d(
            augmented,
            32,
            5,
            (2, 2),
            activation=tf.nn.relu,
            name='conv0')

        # conv1
        conv1 = tf.layers.conv2d(
            conv0,
            32,
            3,
            (2, 2),
            activation=tf.nn.leaky_relu,
            name='conv1')

        # conv2
        conv2 = tf.layers.conv2d(
            conv1,
            32,
            3,
            (1, 1),
            activation=tf.nn.leaky_relu,
            name='conv2')

        # class capsule
        with tf.variable_scope('fully_connected'):
            logits = tf.layers.flatten(conv2)
            logits = tf.layers.dense(logits, 512, activation=tf.nn.leaky_relu)
            logits = tf.layers.dense(logits, self._n_classes)

        return placeholders, logits

    def _loss(self, placeholder_y, logits):
        # one_hot_label = tf.one_hot(placeholder_y, self._n_classes)
        # return tf.losses.hinge_loss(one_hot_label,
        #                             logits)
        if self._loss_fn == 'crossentropy':
            return tf.losses.sparse_softmax_cross_entropy(placeholder_y,
                                                          logits)
        elif self._loss_fn == 'spread':
            # if self._testing:
            #     m = 0.9
            # else:
            #     m = 0.7 * (self._epoch - self._n_epochs) / self._n_epochs \
            #          + 0.2
            m = 0.2
            return sparse_spread_loss(placeholder_y, logits, m)
        elif self._loss_fn == 'seperate_margin':
            return sparse_seperate_margin_loss(placeholder_y, logits)


def _calc_shape(original_shape, stride, kernel_size):
    """
    Helper function that calculate image height and width after convolution.
    """
    shape = [(original_shape[0] - kernel_size) // stride + 1,
             (original_shape[1] - kernel_size) // stride + 1]

    return shape


def augmentate(img, training, testing=False):
    """Do image augmentation in tensorflow.
    """
    def aug_train():
        augged = tf.random_crop(img, [tf.shape(img)[0],
                                      32, 32, 1])
        augged = tf.image.random_brightness(augged, 0.1)
        augged = tf.image.random_contrast(augged, 0.8, 1.2)
        return augged

    def aug_test():
        augged = img[:, 8:40, 8:40, :]
        return augged

    if not testing:
        augged = tf.cond(training, aug_train, aug_test)
    else:
        augged = aug_test()

    augged.set_shape([None, 32, 32, 1])
    return augged
