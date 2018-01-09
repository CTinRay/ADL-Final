import tensorflow as tf
from capsule import conv_capsule, class_capsule
from tf_classifier_base import TFClassifierBase


class CapsuleClassifier(TFClassifierBase):
    def __init__(self, A=32, B=32, C=32, D=32, **kwargs):
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.pose_shape = [4, 4]

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

        # conv0
        conv0 = tf.layers.conv2d(
            placeholders['x'],
            self.A,
            5,
            (2, 2),
            activation=tf.nn.relu,
            name='conv0')

        # primary capsule
        primary_capsule_pose = tf.layers.conv2d(
            conv0,
            self.B * (self.pose_shape[0] * self.pose_shape[1]),
            1,
            (1, 1))
        primary_capsule_pose = tf.reshape(
            primary_capsule_pose,
            tf.shape(primary_capsule_pose)[0],
            primary_capsule_pose.shape[1],
            primary_capsule_pose.shape[2],
            self.B,
            self.pose_shape[0] * self.pose_shape[1],
            name='primary_capsule_pose')

        primary_capsule_active = tf.layers.conv2d(
            conv0,
            self.B * 1,
            1,
            (1, 1),
            activation=tf.nn.sigmoid,
            name='primary_capsule_active')

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
                kernel_size=3,
                stride=1,
                channels_out=self.n_classes)

        logits = class_active
        return placeholders, logits

    def _loss(self, placeholder_y, logits):
        return tf.losses.sparse_softmax_cross_entropy(placeholder_y,
                                                      logits)
