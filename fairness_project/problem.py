import numpy as np


class FairnessProblem:
    def __init__(self, description, x, y, protected_index, fp_weight=0., fn_weight=0.):
        self.description = description
        self.protected_index = protected_index
        self.fp_weight = fp_weight
        self.fn_weight = fn_weight

        self.X = np.array(x)
        self.Y = np.array(y)
        self.X_1 = self.X[np.where(self.X[:, protected_index] == 1)]
        self.Y_1 = self.Y[np.where(self.X[:, protected_index] == 1)]
        self.X_1_pos = self.X_1[np.where(self.Y_1 == 1)]
        self.X_1_neg = self.X_1[np.where(self.Y_1 == 0)]
        self.X_0 = self.X[np.where(self.X[:, protected_index] == 0)]
        self.Y_0 = self.Y[np.where(self.X[:, protected_index] == 0)]
        self.X_0_pos = self.X_0[np.where(self.Y_0 == 1)]
        self.X_0_neg = self.X_0[np.where(self.Y_0 == 0)]

    def set_fp_weight(self, fp_weight):
        self.fp_weight = fp_weight

    def set_fn_weight(self, fn_weight):
        self.fn_weight = fn_weight

