import numpy as np
from sklearn import metrics


class FairnessProblem:
    def __init__(self, description, x, y, protected_index, gamma_gt, gamma_lt, fp_weight=0., fn_weight=0.):
        self.description = description
        self.protected_index = protected_index
        self.fp_weight = fp_weight
        self.fn_weight = fn_weight
        self.gamma_gt = gamma_gt
        self.gamma_lt = gamma_lt
        self.X = np.array(x)
        self.Y = np.array(y)


def split_by_protected_value(x, y, protected_index):
    x_1 = x[np.where(x[:, protected_index] == 1)]
    y_1 = y[np.where(x[:, protected_index] == 1)]
    x_0 = x[np.where(x[:, protected_index] == 0)]
    y_0 = y[np.where(x[:, protected_index] == 0)]
    return x_1, y_1, x_0, y_0


def get_positive_examples(x, y):
    return x[np.where(y == 1)]


def get_negative_examples(x, y):
    return x[np.where(y == 0)]


def measures(y, y_hat):
    _1 = np.ones(y.shape)
    y_hat = y_hat.reshape(y.shape)
    tp = np.sum(np.multiply(y_hat, y))
    tn = np.sum(np.multiply(_1 - y_hat, _1 - y))
    fp = np.sum(np.multiply(y_hat, _1 - y))
    fn = np.sum(np.multiply(_1 - y_hat, y))
    pos = np.sum(y)
    neg = np.sum(_1 - y)

    return {'fpr': fp/neg, 'fnr': fn/pos, 'tpr': tp/pos, 'tnr': tn/neg, 'acc': (tn + tp)/y.shape[0]}


def equalized_odds_reg(x, y, y_hat, protected_index):
    x_1, y_1, x_0, y_0 = split_by_protected_value(x, y, protected_index)
    y_1_hat, y_0_hat = np.array(split_by_protected_value(x, y_hat, protected_index))[[1, 3]]
    x_1_pos = get_positive_examples(x_1, y_1)
    x_0_pos = get_positive_examples(x_0, y_0)
