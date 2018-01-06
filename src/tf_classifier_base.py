import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from batch_generator import BatchGenerator


class TFClassifierBase:

    def _build_model(self):
        """
        return placeholders, logits
        """
        pass

    def _loss(self, _placeholder_y, _logits):
        pass

    def _iter(self, X, y, tensor_loss, train_op, metric_tensors):
        # initialize local variable for metrics
        self._session.run(tf.local_variables_initializer())

        # make accumulator for metric scores
        metric_scores = {}
        for metric in self._metrics:
            metric_scores[metric] = 0

        # make generator
        batch_generator = BatchGenerator(X, y, self._batch_size)

        # run batches for train
        loss = 0
        for b in tqdm(range(X.shape[0] // self._batch_size + 1)):
            batch = next(batch_generator)
            feed_dict = {self._placeholders['x']: batch['x'],
                         self._placeholders['y']: batch['y']}

            if train_op is not None:
                feed_dict[self._placeholders['training']] = True
                batch_loss, _, metrics = self._session.run(
                    [tensor_loss, train_op, metric_tensors],
                    feed_dict=feed_dict)
            else:
                feed_dict[self._placeholders['training']] = False
                batch_loss, metrics \
                    = self._session.run([tensor_loss, metric_tensors],
                                        feed_dict=feed_dict)
            loss += batch_loss

        loss /= X.shape[0] // self._batch_size + 1
        if train_op is None:
            self._history.append(metrics["accuracy"][1])
            if self._early_stop is not None:
                self._history[-1] \
                  = max(self._history[-1:-self._early_stop - 1: -1])

        # put metric score in summary and print them
        epoch_log = {}
        epoch_log['loss'] = float(loss)
        print('loss=%f' % loss)
        for metric in self._metrics:
            score = float(metrics[metric][1])
            epoch_log[metric] = score
            print(', %s=%f' % (metric, score), end='')

        print('\n', end='')
        return epoch_log

    def __init__(self, data_shape, n_classes,
                 learning_rate=1e-3, batch_size=10,
                 n_epochs=10, valid=None,
                 embedding=None, early_stop=None,
                 reg_constant=0.0,
                 gpu_memory_fraction=0.2):

        self._data_shape = data_shape
        self._n_classes = n_classes
        self._optimizer = tf.train.AdamOptimizer(learning_rate)
        self._batch_size = batch_size
        self._n_epochs = n_epochs
        self._metrics = {'accuracy': tf.metrics.accuracy}
        self._global_step = tf.Variable(
            0, dtype=tf.int32, trainable=False, name='global_step')
        self._valid = valid
        self._embedding = embedding
        self._history = []
        self._early_stop = early_stop
        self._reg_constant = reg_constant
        self._epoch = 0

        # limit GPU memory usage
        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=gpu_memory_fraction)
        self._session = tf.Session(
            config=tf.ConfigProto(gpu_options=gpu_options))

        # build graph
        with tf.variable_scope('TF-Classifier') as scope:
            self._placeholders, self._logits \
              = self._build_model()

    def fit(self, X, y, callbacks=[]):
        # make loss tensor
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss = self._loss(self._placeholders['y'], self._logits) # \
            # + tf.reduce_sum(reg_losses) * self._reg_constant

        # make train operator
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = self._optimizer.minimize(
                loss, global_step=self._global_step)

        # make metric tensors
        metric_tensors = {}
        for metric in self._metrics:
            y_pred_argmax = tf.argmax(self._logits, axis=-1)
            # y_true_argmax = tf.argmax(self._placeholders['y'], axis=-1)
            mask = tf.cast(self._placeholders['y'] >= 0, tf.float32)
            metric_tensors[metric] = \
                self._metrics[metric](self._placeholders['y'],
                                      y_pred_argmax,
                                      weights=mask)

        # initialize
        self._session.run(tf.global_variables_initializer())

        # Start the training loop.
        while self._epoch < self._n_epochs:
            # train and evaluate train score
            print('training %i' % self._epoch)
            log_train = self._iter(X, y, loss, train_op,
                                   metric_tensors)

            # evaluate valid score
            if self._valid is not None:
                print('evaluating %i' % self._epoch)
                log_valid = self._iter(self._valid['x'],
                                       self._valid['y'], loss, None,
                                       metric_tensors)
            else:
                log_valid = None

            for callback in callbacks:
                callback.on_epoch_end(log_train, log_valid, self)

            self._epoch += 1

    def predict_prob(self, X, batch_size=24):
        y_prob = None
        with tf.variable_scope('nn', reuse=True):
            for batch in BatchGenerator(X, None, batch_size, shuffle=False):
                batch_y_prob = self._session.run(
                    self._logits,
                    feed_dict={self._placeholders['x']: batch['x'],
                               self._placeholders['training']: False})
                if y_prob is None:
                    y_prob = batch_y_prob
                else:
                    y_prob = np.concatenate([y_prob, batch_y_prob],
                                            axis=0)

        return y_prob

    def predict(self, X, batch_size=24):
        prob = self.predict_prob(X)
        return np.argmax(prob, axis=-1)

    def save(self, path):
        saver = tf.train.Saver()
        filename = os.path.join(path, 'model.ckpt')
        saver.save(self._session, filename)

    def load(self, path):
        saver = tf.train.Saver()
        filename = os.path.join(path, 'model.ckpt')
        saver.restore(self._session, filename)
