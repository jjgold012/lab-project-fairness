import numpy as np
from sklearn import metrics


class FairnessProblem:
    def __init__(self,
                 description,
                 x,
                 y,
                 protected_index,
                 gamma_gt=0,
                 gamma_lt=4,
                 weight_gt=0,
                 weight_lt=150,
                 fp=True,
                 fn=True,
                 weight_res=3,
                 gamma_res=1,
                 test_size=0.33,
                 num_of_tries=5
                 ):
        self.description = description
        self.protected_index = protected_index
        self.weight_gt = weight_gt
        self.weight_lt = weight_lt
        self.fp = fp
        self.fn = fn
        self.gamma_gt = gamma_gt
        self.gamma_lt = gamma_lt
        self.weight_res = weight_res
        self.gamma_res = gamma_res
        self.test_size = test_size
        self.X = np.array(x)
        self.Y = np.array(y)
        self.num_of_tries = num_of_tries


class Results:
    def __init__(self, w, ll, fnr_relaxed_diff, fpr_relaxed_diff, objective):
        self.w = w
        self.ll = ll
        self.fnr_relaxed_diff = fnr_relaxed_diff
        self.fpr_relaxed_diff = fpr_relaxed_diff
        self.objective = objective


class Measures:
    def __init__(self, acc, _1_acc, _1_fpr, _1_fnr, _0_acc, _0_fpr, _0_fnr, fpr_diff, fnr_diff):
        self.acc = acc
        self._1_acc = _1_acc
        self._1_fpr = _1_fpr
        self._1_fnr = _1_fnr
        self._0_acc = _0_acc
        self._0_fpr = _0_fpr
        self._0_fnr = _0_fnr
        self.fpr_diff = fpr_diff
        self.fnr_diff = fnr_diff


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
    tp = y_hat.dot(y)
    tn = (_1 - y_hat).dot(_1 - y)
    fp = y_hat.dot(_1 - y)
    fn = (_1 - y_hat).dot(y)
    pos = np.sum(y)
    neg = np.sum(_1 - y)

    return {'fpr': fp/neg, 'fnr': fn/pos, 'tpr': tp/pos, 'tnr': tn/neg, 'acc': (tn + tp)/y.shape[0]}


def create_synthetic_problem(epsilon=0.125):
    print("epsilon: " + str(epsilon))

    x = list()
    y = list()
    for i in range(5000):
        new_x = list()
        new_y = float(np.random.randint(2))
        x_1 = new_y if np.random.rand() >= epsilon else 1 - new_y
        x_2 = new_y if np.random.rand() >= 2*epsilon else 1 - new_y

        new_x.append(1.)
        new_x.append(x_1)
        new_x.append(x_2)
        y.append(new_y)
        x.append(new_x)
    return FairnessProblem(
        description="synthetic data with epsilon: " + str(epsilon),
        x=x,
        y=y,
        protected_index=1
    )

