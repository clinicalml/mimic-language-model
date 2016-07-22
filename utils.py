import itertools

import numpy as np


class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'


def print_color(s, color=None):
    if color:
        print color + str(s) + Colors.ENDC,
    else:
        print s,


def grouper(n, iterable, fillvalue=None):
    args = [iter(iterable)] * n
    return itertools.izip_longest(*args, fillvalue=fillvalue)


def inspection_decide_color(diff):
    if diff < -0.1:
        return Colors.FAIL
    elif diff > 0.2:
        return Colors.OKGREEN
    elif diff > 0.1:
        return Colors.WARNING
    else:
        return None


def inspect_conditional_utility(xs, ms, differences, config, vocab):
    X = np.concatenate(xs, 1)
    M = np.concatenate([np.ones([config.batch_size, 1])] + ms, 1)
    diffs = np.concatenate([np.zeros([config.batch_size, 1])] + differences, 1)
    for i in range(config.batch_size):
        print
        for j in range(len(X[i])):
            if not M[i,j]: break
            print_color(vocab.vocab_list[X[i,j]].ljust(len("%.2f" % diffs[i,j])),
                        inspection_decide_color(diffs[i,j]))
        print
        for j in range(len(X[i])):
            if not M[i,j]: break
            print_color(("%.2f" % diffs[i,j]).ljust(len(vocab.vocab_list[X[i,j]])),
                        inspection_decide_color(diffs[i,j]))
        print
    print
