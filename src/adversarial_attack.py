import argparse
import os
import pdb
import sys
import traceback
import numpy as np
import tensorflow as tf
from capsule_classifier import CapsuleClassifier
from matrix_cnn_classifier import MatrixCNNClassifier
from cnn_classifier import CNNClassifier
from callbacks import ModelCheckpoint, LogCallback
from dataset import SmallNORBDataset


def adversarial_attack(clf, data,
                       epsilon_max, epsilon_step,
                       method='FGSM', batch_size=128):
    """Do adversarial attack on the classifier and data.
    This function maximizes the loss of the classifier over the data
    increamentally and return the corresponding accuracy of each step.

    Args:
        clf (TFClassifierBase): Classifier to be attacked.
        data (dict): Dictionary which has key 'x', 'y'.
        epsilon_max (float): Max epsilon to test.
        epsilon_step (float): Size of epsilon.
        method (str): Method of adversarial attack. Currently supported method
            include 'FGSM', 'BIM'.

    Returns:
        accuracy (list): List of float of epsilon_max // epsilon_step elements.
    """
    # create tensorflow op to calculate gradient
    loss = clf._loss(clf._placeholders['y'], clf._logits)
    op_gradients = tf.gradients(tf.reduce_sum(loss), clf._placeholders['x'])[0]
    op_gradients = tf.sign(op_gradients)
    op_gradients *= epsilon_step

    # modify data['x'] to x_ to do attack
    x_ = np.array(data['x'])

    # list of accuracy of each steps
    accuracy = []

    # place to store gradient
    gradients = np.zeros_like(data['x'])

    # run steps
    for step in range(int(epsilon_max / epsilon_step)):
        # calculate accuracy over x_
        y_ = clf.predict(x_)
        accuracy.append(np.mean(y_ == data['y']))
        print('step={}, epsilon={}, accuracy={}'
              .format(step, step * epsilon_step, accuracy[-1]))

        # for FGSM, only calculate gradients for the first step.
        # for FGSM, update gradients for each step.
        if method == 'BIM' or step == 0:
            for b in range(0, data['x'].shape[0], batch_size):
                gradients[b: b + batch_size] = clf._session.run(op_gradients, {
                    clf._placeholders['x']: x_[b: b + batch_size],
                    clf._placeholders['y']: data['y'][b: b + batch_size]})

        x_ += gradients

    return accuracy


def main(args):
    smallnorb = SmallNORBDataset(
        None, os.path.join(args.data_dir, 'test.npz'))

    test = smallnorb.get_test()

    n_classes = smallnorb.get_n_classes()
    if args.arch == 'matrix_capsule':
        classifier = CapsuleClassifier(test['x'].shape[1:],
                                       n_classes,
                                       batch_size=args.batch_size,
                                       n_epochs=100)
    elif args.arch == 'matrix_cnn':
        classifier = MatrixCNNClassifier(test['x'].shape[1:],
                                         n_classes,
                                         batch_size=args.batch_size,
                                         n_epochs=100,
                                         testing=True)
    elif args.arch == 'cnn':
        classifier = CNNClassifier(test['x'].shape[1:],
                                   n_classes,
                                   batch_size=args.batch_size,
                                   n_epochs=100,
                                   testing=True)

    classifier.load(args.ckp_path)
    accuracy = adversarial_attack(classifier, test,
                                  args.epsilon_max,
                                  args.epsilon_step,
                                  args.method,
                                  args.batch_size)

    # dump accuracy to file
    with open(args.output, 'w') as f:
        for i, a in enumerate(accuracy):
            f.write('{},{}\n'
                    .format(i * args.epsilon_step, a))


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Script to experient with adversarial attack.")
    parser.add_argument('data_dir', type=str,
                        help='Directory that contains train.npz and'
                        'test.nzp generated with scripts/smallnorb2npz.py')
    parser.add_argument('ckp_path', type=str,
                        help='Path to store checkpoint and log.')
    parser.add_argument('output', type=str,
                        help='Destination file to dump accuracy log.')
    parser.add_argument('--arch', type=str, default='matrix_cnn',
                        help='Architecture of network. Currently suport'
                        'matrix_capsule, matrix_cnn, cnn')
    parser.add_argument('--epsilon_max', type=float, default=0.5,
                        help='Max epsilon to experient')
    parser.add_argument('--epsilon_step', type=float, default=0.01,
                        help='Step size of epsilon')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for calculate epsilon.')
    parser.add_argument('--method', type=str, default='FGSM',
                        help='Method of adversarial attack. Currently'
                             'support FGSM and BIM')
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
