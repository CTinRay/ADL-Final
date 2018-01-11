import argparse
import os
import pdb
import sys
import traceback
import numpy as np
from capsule_classifier import CapsuleClassifier
from dataset import SmallNORBDataset


def main(args):
    smallnorb = SmallNORBDataset(
        os.path.join(args.data_dir, 'train.npz'),
        os.path.join(args.data_dir, 'test.npz'))

    train, valid = smallnorb.get_train_valid()

    classifier = CapsuleClassifier(train['x'].shape[1:],
                                   np.max(train['y'] + 1),
                                   valid=valid)
    classifier.fit(train['x'], train['y'])


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Transform SmallNORB dataset to npz")
    parser.add_argument('data_dir', type=str,
                        help='Directory that contains train.npz and'
                        'test.nzp generated with scripts/smallnorb2npz.py')
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
