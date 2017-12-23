import cvxpy as cp
import numpy as np
from pprint import pprint
from sklearn.model_selection import train_test_split
from fairness_project.fairness_data import *


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def solve_non_convex():
    print('non-convex')


def solve_convex(x, y, protected_index, gamma, fp_weight, fn_weight, squared=True):
    x_1, y_1, x_0, y_0 = split_by_protected_value(x, y, protected_index)
    x_1_pos = get_positive_examples(x_1, y_1)
    x_1_neg = get_negative_examples(x_1, y_1)
    x_0_pos = get_positive_examples(x_0, y_0)
    x_0_neg = get_negative_examples(x_0, y_0)
    w = cp.Variable(x.shape[1])

    diff_pos = (np.sum(x_1_pos, axis=0)/x_1_pos.shape[0]) - (np.sum(x_0_pos, axis=0)/x_0_pos.shape[0])
    diff_neg = (np.sum(x_1_neg, axis=0)/x_1_neg.shape[0]) - (np.sum(x_0_neg, axis=0)/x_0_neg.shape[0])

    log_likelihood = (y.T*(x*w) - cp.sum_entries(cp.logistic(x*w)))
    
    if squared:
        fpr_diff = \
            cp.square(diff_neg*w)
        fnr_diff = \
            cp.square(diff_pos*w)
    else:
        fpr_diff = \
            cp.abs(diff_neg*w)
        fnr_diff = \
            cp.abs(diff_pos*w)

    w_norm_square = cp.sum_squares(w)

    objective = cp.Minimize(-log_likelihood +
                            fn_weight*fnr_diff +
                            fp_weight*fpr_diff +
                            gamma*w_norm_square)

    p = cp.Problem(objective)
    p.solve()

    return {
        'w': w.value,
        'll': log_likelihood.value,
        'fnr_diff': fnr_diff.value,
        'fpr_diff': fpr_diff.value
    }


def measure_results(x_test, y_test, protected_index, w):
    y_hat = np.round(sigmoid(np.dot(x_test, w)))
    y_1, y_0 = np.array(split_by_protected_value(x_test, y_test, protected_index))[[1, 3]]
    y_1_hat, y_0_hat = np.array(split_by_protected_value(x_test, y_hat, protected_index))[[1, 3]]

    acc_measures = dict()
    acc_measures['all'] = measures(y_test, y_hat)
    acc_measures['1'] = measures(y_1, y_1_hat)
    acc_measures['0'] = measures(y_0, y_0_hat)

    return {
        'acc': acc_measures['all']['acc'],
        'fpr': acc_measures['all']['fnr'],
        'fnr': acc_measures['all']['fpr'],
        '1_acc': acc_measures['1']['acc'],
        '1_fpr': acc_measures['1']['fnr'],
        '1_fnr': acc_measures['1']['fpr'],
        '0_acc': acc_measures['0']['acc'],
        '0_fpr': acc_measures['0']['fnr'],
        '0_fnr': acc_measures['0']['fpr'],
        'fpr_diff': abs(acc_measures['1']['fpr'] - acc_measures['0']['fpr']),
        'fnr_diff': abs(acc_measures['1']['fnr'] - acc_measures['0']['fnr'])
    }


def fairness(problem: FairnessProblem):
    x = problem.X
    y = problem.Y
    fp_weight = problem.fp_weight
    fn_weight = problem.fn_weight

    protected_index = problem.protected_index

    results_squared = list()
    results_abs = list()
    for gamma in np.linspace(problem.gamma_gt, problem.gamma_lt, num=10):
        temp_res_squared = list()
        temp_res_abs = list()
        for j in range(10):
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=j)

            res_squared = solve_convex(x_train, y_train, protected_index, gamma, fp_weight=fp_weight, fn_weight=fn_weight, squared=True)
            res_abs = solve_convex(x_train, y_train, protected_index, gamma, fp_weight=fp_weight, fn_weight=fn_weight, squared=False)

            measures_squared = measure_results(x_test, y_test, protected_index, res_squared['w'])
            measures_abs = measure_results(x_test, y_test, protected_index, res_abs['w'])

            temp_res_squared.append(measures_squared)
            temp_res_abs.append(measures_abs)

        gamma_result_squared = dict()
        gamma_result_abs = dict()
        gamma_result_squared['gamma'] = gamma
        gamma_result_abs['gamma'] = gamma
        for key in temp_res_squared[0].keys():
            gamma_result_squared[key] = np.average(np.array([r[key] for r in temp_res_squared]))
            gamma_result_abs[key] = np.average(np.array([r[key] for r in temp_res_abs]))

        results_squared.append(gamma_result_squared)
        results_abs.append(gamma_result_abs)

        pprint(gamma_result_squared)

# equalized_odds_reg(x_train,y_train,y_train,problem.protected_index)


