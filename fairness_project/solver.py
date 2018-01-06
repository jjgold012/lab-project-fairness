import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
from sklearn.model_selection import train_test_split
from fairness_project.fairness_data import *


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def plot_results(subplot, results, type):
    weights = [r['weight'] for r in results]
    acc = [r['measures']['acc'] for r in results]
    fnr_diff = [r['measures']['fnr_diff'] for r in results]
    fpr_diff = [r['measures']['fpr_diff'] for r in results]
    r_fnr_diff = [r['results']['fnr_diff'] for r in results]
    r_fpr_diff = [r['results']['fpr_diff'] for r in results]
    subplot.set_title(type)
    subplot.plot(weights, acc, 'r-', label="Accuracy", linewidth=2)
    subplot.plot(weights, fnr_diff, 'g-', label="fnr diff", linewidth=2)
    subplot.plot(weights, r_fnr_diff, 'g--', label="relaxed fnr diff", linewidth=2)
    subplot.plot(weights, fpr_diff, 'b-', label="fpr diff", linewidth=2)
    subplot.plot(weights, r_fpr_diff, 'b--', label="relaxed fpr diff", linewidth=2)
    subplot.legend(loc='best', prop={'size':11}, ncol=1)


def show_results(results_squared, results_abs):
    fig = plt.figure(figsize=(10, 5))
    print('The result for squared relaxation')
    pprint(results_squared)
    sub1 = fig.add_subplot(121)
    plot_results(sub1, results_squared, type='Squared')

    print('The result for absolute value relaxation')
    pprint(results_abs)
    sub2 = fig.add_subplot(122)
    plot_results(sub2, results_abs, type='Absolute value')
    plt.show()


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
        'fpr_diff': fpr_diff.value,
        'objective': objective.value
    }


def measure_results(x_test, y_test, protected_index, w, fp, fn):
    y_hat = np.round(sigmoid(np.dot(x_test, w)))
    y_1, y_0 = np.array(split_by_protected_value(x_test, y_test, protected_index))[[1, 3]]
    y_1_hat, y_0_hat = np.array(split_by_protected_value(x_test, y_hat, protected_index))[[1, 3]]

    all_measures = measures(y_test, y_hat)
    _1_measures = measures(y_1, y_1_hat)
    _0_measures = measures(y_0, y_0_hat)
    fpr_diff = abs(_1_measures['fpr'] - _0_measures['fpr'])
    fnr_diff = abs(_1_measures['fnr'] - _0_measures['fnr'])
    return {
        'acc': all_measures['acc'],
        '1_fpr': _1_measures['fpr'],
        '1_fnr': _1_measures['fnr'],
        '0_fpr': _0_measures['fpr'],
        '0_fnr': _0_measures['fnr'],
        'fpr_diff': fpr_diff,
        'fnr_diff': fnr_diff,
        'objective': all_measures['acc'] + (fpr_diff if fp else 0) + (fnr_diff if fn else 0)
    }


def fairness(problem: FairnessProblem):
    x = problem.X
    y = problem.Y
    protected_index = problem.protected_index
    print(problem.description)
    results_squared = list()
    results_abs = list()
    for weight in np.linspace(problem.weight_gt, problem.weight_lt, num=problem.weight_res):
        print('weight:', weight)
        res_squared = list()
        res_abs = list()
        fp_weight = float(weight if problem.fp else 0)
        fn_weight = float(weight if problem.fn else 0)
        for gamma in np.linspace(problem.gamma_gt, problem.gamma_lt, num=problem.gamma_res):
            print('gamma:', gamma)
            temp_res_squared = list()
            temp_res_abs = list()
            for j in range(5): #TODO check for cross validation sklearn
                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=j)

                conv_squared = solve_convex(x_train, y_train, protected_index, gamma, fp_weight=fp_weight, fn_weight=fn_weight, squared=True)
                conv_abs = solve_convex(x_train, y_train, protected_index, gamma, fp_weight=fp_weight, fn_weight=fn_weight, squared=False)
                measures_squared = measure_results(x_test, y_test, protected_index, conv_squared['w'], fp=problem.fp, fn=problem.fn)
                measures_abs = measure_results(x_test, y_test, protected_index, conv_abs['w'], fp=problem.fp, fn=problem.fn)

                temp_res_squared.append({'results': conv_squared, 'measures': measures_squared})
                temp_res_abs.append({'results': conv_abs, 'measures': measures_abs})

            gamma_result_squared = {'gamma': gamma}
            gamma_result_abs = {'gamma': gamma}
            for key1 in temp_res_squared[0].keys():
                gamma_result_squared[key1] = dict()
                gamma_result_abs[key1] = dict()
                for key2 in temp_res_squared[0][key1].keys():
                    gamma_result_squared[key1][key2] = np.average(np.array([r[key1][key2] for r in temp_res_squared]))
                    gamma_result_abs[key1][key2] = np.average(np.array([r[key1][key2] for r in temp_res_abs]))

            res_squared.append(gamma_result_squared)
            res_abs.append(gamma_result_abs)

        best_squared = res_squared[np.array([r['measures']['objective'] for r in res_squared]).argmin()]
        best_abs = res_abs[np.array([r['measures']['objective'] for r in res_abs]).argmin()]
        best_squared['weight'] = weight
        best_abs['weight'] = weight
        results_squared.append(best_squared)
        results_abs.append(best_abs)

    show_results(results_squared, results_abs)


