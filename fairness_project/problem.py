import numpy as np


class FairnessProblem:
    def __init__(self, description, x, y, protected_index):
        self.description = description
        self.protected_index = protected_index
        self.X = np.array(x)
        self.Y = np.array(y)
        self.X_1 = self.X[np.where(self.X[:, protected_index] == 1)]
        self.Y_1 = self.Y[np.where(self.X[:, protected_index] == 1)]
        self.X_0 = self.X[np.where(self.X[:, protected_index] == 0)]
        self.Y_0 = self.Y[np.where(self.X[:, protected_index] == 0)]
        print(self.X_1.shape)


