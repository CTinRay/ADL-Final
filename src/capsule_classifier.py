import pdb
import tensorflow as tf
from capsule import conv_capsule, class_capsule
from tf_classifier_base import TFClassifierBase


class CapsuleClassifier(TFClassifierBase):
    def __init__(self, *args, A=8, B=8, C=8, D=8, **kwargs):
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.pose_shape = [4, 4]
        super(CapsuleClassifier, self).__init__(*args, **kwargs)

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
            self.A,
            5,
            (2, 2),
            activation=tf.nn.relu,
            name='conv0')

        # primary capsule
        with tf.variable_scope('primary_capsule'):
            primary_capsule_pose = tf.layers.conv2d(
                conv0,
                self.B * (self.pose_shape[0] * self.pose_shape[1]),
                1,
                (1, 1))
            img_shape = _calc_shape(img_shape, stride=2, kernel_size=5)

            primary_capsule_pose = tf.reshape(
                primary_capsule_pose,
                [tf.shape(primary_capsule_pose)[0],
                 img_shape[0],
                 img_shape[1],
                 self.B,
                 self.pose_shape[0],
                 self.pose_shape[1]],
                name='pose')

            primary_capsule_active = tf.layers.conv2d(
                conv0,
                self.B * 1,
                1,
                (1, 1),
                activation=tf.nn.sigmoid,
                name='activation')

        # capsule1
        with tf.variable_scope('capsule1'):
            conv_capsule1_pose, conv_capsule1_active = conv_capsule(
                primary_capsule_pose,
                primary_capsule_active,
                kernel_size=3,
                stride=2,
                channels_out=self.C)

        # capsule2
        with tf.variable_scope('capsule2'):
            conv_capsule2_pose, conv_capsule2_active = conv_capsule(
                conv_capsule1_pose,
                conv_capsule1_active,
                kernel_size=3,
                stride=1,
                channels_out=self.D)

        # class capsule
        with tf.variable_scope('class_capsule'):
            class_pose, class_active = class_capsule(
                conv_capsule2_pose,
                conv_capsule2_active,
                n_classes=int(self._n_classes),
                routing_iters=1)

        logits = class_active
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
