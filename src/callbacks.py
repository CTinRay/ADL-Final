import os
import math
import numpy as np


class Callback:
    def __init__():
        pass

    def on_epoch_end(log_train, log_valid, model):
        pass


class ModelCheckpoint(Callback):
    def __init__(self, filepath,
                 monitor='loss',
                 verbose=0,
                 mode='min'):
        self._filepath = filepath
        self._verbose = verbose
        self._monitor = monitor
        self._best = math.inf if mode == 'min' else - math.inf
        self._mode = mode

    def on_epoch_end(self, log_train, log_valid, model):
        score = log_valid[self._monitor]
        if self._mode == 'all':
            model.save(self._filepath)
            if self._verbose > 0:
                print('Model saved (%f)' % score)
        elif self._mode == 'min':
            if score < self._best:
                self._best = score
                model.save(self._filepath)
                if self._verbose > 0:
                    print('Best model saved (%f)' % score)
        elif self._mode == 'max':
            if score > self._best:
                self._best = score
                model.save(self._filepath)
                if self._verbose > 0:
                    print('Best model saved (%f)' % score)

class LogCallback(Callback):
    def __init__(self, path):
        self._path = path
        self._fp_train_acc = open(os.path.join(path, 'train-accuracy.log'),
                                  'w', buffering=1)
        self._fp_train_loss = open(os.path.join(path, 'train-loss.log'),
                                   'w', buffering=1)
        self._fp_valid_acc = open(os.path.join(path, 'valid-accuracy.log'),
                                  'w', buffering=1)
        self._fp_valid_loss = open(os.path.join(path, 'valid-loss.log'),
                                   'w', buffering=1)

    def on_epoch_end(self, log_train, log_valid, model):
        self._fp_train_acc.write('{},{}\n'
                                 .format(model._epoch, log_train['accuracy']))
        self._fp_train_loss.write('{},{}\n'
                                  .format(model._epoch, log_train['loss']))
        self._fp_valid_acc.write('{},{}\n'
                                 .format(model._epoch, log_valid['accuracy']))
        self._fp_valid_loss.write('{},{}\n'
                                  .format(model._epoch, log_valid['loss']))
