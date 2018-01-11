"""
Make sure you have
https://github.com/ndrplz/small_norb/blob/master/smallnorb/dataset.py
in scripts/ folder.
"""
import argparse
import os
import pdb
import sys
import traceback
import numpy as np
from dataset import SmallNORBDataset


def data2np(data):
    image_lt = []
    image_rt = []
    categories = []
    for exp in data:
        image_lt.append(exp.image_lt)
        image_rt.append(exp.image_rt)
        categories.append(exp.category)

    image_lt = np.array(image_lt)
    image_rt = np.array(image_rt)
    categories = np.array(categories)

    return image_lt, image_rt, categories


def main(args):
    smallnorb = SmallNORBDataset(args.data_dir)

    image_lt, image_rt, categories = data2np(smallnorb.data['train'])
    np.savez_compressed(os.path.join(args.dest_dir, 'train.npz'),
                        lt=image_lt, rt=image_rt, category=categories)

    image_lt, image_rt, categories = data2np(smallnorb.data['test'])
    np.savez_compressed(os.path.join(args.dest_dir, 'test.npz'),
                        lt=image_lt, rt=image_rt, category=categories)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Transform SmallNORB dataset to npz")
    parser.add_argument('data_dir', type=str,
                        help='Directory that contains training and'
                        'testing file downloaded from'
                        'https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/')
    parser.add_argument('dest_dir', type=str,
                        help='Directory of the output file.')
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
