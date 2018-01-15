import tensorflow as tf


def sparse_spread_loss(labels, logits, m=0.2):
    one_hot_label = tf.one_hot(labels, logits.shape[-1])
    a_t = tf.reduce_sum(logits * one_hot_label, -1, keep_dims=True)
    max_square = tf.maximum(0.0, m - (a_t - logits)) ** 2
    loss = tf.reduce_sum(max_square * (1 - one_hot_label))
    return loss


def sparse_seperate_margin_loss(labels, logits,
                                m_plus=0.9, m_minus=0.1,
                                down_weight=0.5):
    t = tf.one_hot(labels, logits.shape[-1])
    loss = t * tf.maximum(0.0, m_plus - logits) ** 2 \
        + down_weight * (1 - t) * tf.maximum(0.0, logits - m_minus) ** 2
    loss = tf.reduce_sum(loss)
    return loss
