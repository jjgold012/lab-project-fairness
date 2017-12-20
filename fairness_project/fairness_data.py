import numpy as np


class FairnessProblem:
    def __init__(self, description, x, y, protected_index, fp_fn_weight=0.):
        self.description = description
        self.protected_index = protected_index
        self.fp_fn_weight = fp_fn_weight

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
    y_hat = y_hat.reshape(y.shape)
    _1 = np.ones(y.shape, dtype=np.int)

    tp = np.sum(np.multiply(y_hat, y))
    fp = np.sum(np.multiply(y_hat, _1 - y))
    tn = np.sum(np.multiply(_1 - y_hat, _1 - y))
    fn = np.sum(np.multiply(_1 - y_hat, y))
    # pos = np.sum(y_hat)
    # neg = np.sum(_1 - y_hat)

    # return [float(TP)/(TP+FN), float(FP)/(FP+TN), float(TN)/(TN+FP), float(FN)/(FN+TP)]
    return [tp/(tp+fn), fp/(fp+tn), tn/(tn+fp), fn/(fn+tp)]


def equalized_odds_reg(x, y, y_hat, protected_index):
    x_1, y_1, x_0, y_0 = split_by_protected_value(x, y, protected_index)
    y_1_hat, y_0_hat = np.array(split_by_protected_value(x, y_hat, protected_index))[[1, 3]]
    x_1_pos = get_positive_examples(x_1, y_1)
    x_0_pos = get_positive_examples(x_0, y_0)
