import numpy as np


class FairnessProblem:
    def __init__(self, description, x, y, protected_index, fp_weight=0., fn_weight=0.):
        self.description = description
        self.protected_index = protected_index
        self.fp_weight = fp_weight
        self.fn_weight = fn_weight

        self.X = np.array(x)
        self.Y = np.array(y)
        self.X_1, self.Y_1, self.X_0, self.Y_0 = split_by_protected_value(self.X, self.Y, self.protected_index)

        self.X_1_pos = get_positive_examples(self.X_1, self.Y_1)
        self.X_1_neg = get_negative_examples(self.X_1, self.Y_1)
        self.X_0_pos = get_positive_examples(self.X_0, self.Y_0)
        self.X_0_neg = get_negative_examples(self.X_0, self.Y_0)

    def set_fp_weight(self, fp_weight):
        self.fp_weight = fp_weight

    def set_fn_weight(self, fn_weight):
        self.fn_weight = fn_weight


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

