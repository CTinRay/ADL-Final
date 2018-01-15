import argparse
import os
import pdb
import sys
import traceback
import numpy as np
from IPython.terminal.embed import embed
from capsule_classifier import CapsuleClassifier
from matrix_cnn_classifier import MatrixCNNClassifier
from cnn_classifier import CNNClassifier
from callbacks import ModelCheckpoint, LogCallback
from dataset import SmallNORBDataset


def main(args):
    smallnorb = SmallNORBDataset(
        os.path.join(args.data_dir, 'train.npz'),
        os.path.join(args.data_dir, 'test.npz'))

    train, valid = smallnorb.get_train_valid()

    if args.arch == 'matrix_capsule':
        classifier = CapsuleClassifier(train['x'].shape[1:],
                                       np.max(train['y']) + 1,
                                       valid=valid,
                                       batch_size=args.batch_size,
                                       n_epochs=100)
    elif args.arch == 'matrix_cnn':
        classifier = MatrixCNNClassifier(train['x'].shape[1:],
                                         np.max(train['y']) + 1,
                                         valid=valid,
                                         batch_size=args.batch_size,
                                         n_epochs=100)
    elif args.arch == 'cnn':
        classifier = CNNClassifier(train['x'].shape[1:],
                                   np.max(train['y']) + 1,
                                   valid=valid,
                                   batch_size=args.batch_size,
                                   n_epochs=100,
                                   loss_fn=args.loss)

    model_checkpoint = ModelCheckpoint(args.ckp_path,
                                       'loss', 1, 'all')
    log_callback = LogCallback(args.ckp_path)
    classifier.fit(train['x'], train['y'],
                   callbacks=[model_checkpoint, log_callback])
    print(classifier.predict_prob(train['x']))


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Script to train model.")
    parser.add_argument('data_dir', type=str,
                        help='Directory that contains train.npz and'
                        'test.nzp generated with scripts/smallnorb2npz.py')
    parser.add_argument('ckp_path', type=str,
                        help='Path to store checkpoint and log')
    parser.add_argument('--arch', type=str, default='matrix_cnn',
                        help='Architecture of network. Currently support'
                        'matrix_capsule, matrix_cnn, cnn')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for training.')
    parser.add_argument('--loss', type=str, default='crossentropy',
                        help='Loss function to use. Currently support loss'
                        'functions are crossentropy, seperate_margin, spread')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = _parse_args()
    try:
        main(args)
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
