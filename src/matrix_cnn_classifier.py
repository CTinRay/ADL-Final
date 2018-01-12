import pdb
import tensorflow as tf
from matrix_cnn import matrix_cnn
from tf_classifier_base import TFClassifierBase


class MatrixCNNClassifier(TFClassifierBase):
    def __init__(self, *args, **kwargs):
        self.pose_shape = [16, 16]
        super(MatrixCNNClassifier, self).__init__(*args, **kwargs)

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

        augmented = augmentate(placeholders['x'], placeholders['training'])
        img_shape = [32, 32, 1]

        # conv0
        conv0 = tf.layers.conv2d(
            augmented,
            32,
            5,
            (2, 2),
            activation=tf.nn.relu,
            name='conv0')

        # primary capsule
        with tf.variable_scope('primary_cnn'):
            primary_cnn_matrix = tf.layers.conv2d(
                conv0,
                (self.pose_shape[0] * self.pose_shape[1]),
                1,
                (1, 1))
            img_shape = _calc_shape(img_shape, stride=2, kernel_size=5)

            primary_cnn_matrix = tf.reshape(
                primary_cnn_matrix,
                [tf.shape(primary_cnn_matrix)[0],
                 img_shape[0],
                 img_shape[1],
                 self.pose_shape[0],
                 self.pose_shape[1]],
                name='pose')

        # capsule1
        with tf.variable_scope('matrix_cnn1'):
            cnn1_matrix = matrix_cnn(
                primary_cnn_matrix,
                kernel_size=3,
                stride=2)

        # capsule2
        with tf.variable_scope('matrix_cnn2'):
            cnn2_matrix = matrix_cnn(
                cnn1_matrix,
                kernel_size=3,
                stride=1)

        # class capsule
        with tf.variable_scope('fully_connected'):
            logits = tf.reshape(cnn2_matrix, [tf.shape(cnn2_matrix)[0],
                                              cnn2_matrix.shape[1]
                                              * cnn2_matrix.shape[2]
                                              * self.pose_shape[0]
                                              * self.pose_shape[1]])
            logits = tf.layers.dense(logits, 512, activation=tf.nn.leaky_relu)
            logits = tf.layers.dense(logits, self._n_classes)

        return placeholders, logits

    def _loss(self, placeholder_y, logits):
        one_hot_label = tf.one_hot(placeholder_y, self._n_classes)
        return tf.losses.hinge_loss(one_hot_label,
                                    logits)


def _calc_shape(original_shape, stride, kernel_size):
    """
    Helper function that calculate image height and width after convolution.
    """
    shape = [(original_shape[0] - kernel_size) // stride + 1,
             (original_shape[1] - kernel_size) // stride + 1]

    return shape


def augmentate(img, training):
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

    augged = tf.cond(training, aug_train, aug_test)
    augged.set_shape([None, 32, 32, 1])

    return augged
