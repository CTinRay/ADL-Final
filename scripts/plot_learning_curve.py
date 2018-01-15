import pdb
import argparse
from matplotlib import pyplot as plt
import numpy as np


def read_file(filename):
    xs = []
    ys = []
    with open(filename) as f:
        for l in f:
            cols = l.split(',')
            xs.append(float(cols[0]))
            ys.append(float(cols[1]))

    return xs, ys


def main(args):
    fig = plt.figure(dpi=450)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel(args.x_title)
    ax.set_ylabel(args.y_title)

    # ax.set_ylim([0, 1])
    ax.grid()

    colors = ['crimson', 'orange', 'blue', 'c', 'm', 'y']
    for i in range(len(args.log_file)):
        xs, ys = read_file(args.log_file[i])

        label = args.labels[i] if args.labels is not None else None
        ax.plot(xs, ys,
                color=colors[i // 2],
                linestyle='dashdot' if i % 2 == 0 else 'solid',
                label=label, linewidth=2.0)

    ax.legend()
    fig.savefig(args.output)


def parse_args():
    parser = argparse.ArgumentParser(description='Plot learning curve.')
    parser.add_argument('log_file', type=str, nargs='+',
                        help='Path to the log file.')
    parser.add_argument('output', type=str,
                        help='Directory of the data.')
    parser.add_argument('--labels', type=str, nargs='+', default=None)
    parser.add_argument('--x_title', type=str, default='')
    parser.add_argument('--y_title', type=str, default='')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
