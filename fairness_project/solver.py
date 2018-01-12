import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
from sklearn.model_selection import train_test_split
from fairness_data import *
# import gurobipy


def plot_results(subplot, results, type):
    weights = [r['weight'] for r in results]
    acc = [r['test_measures']['acc'] for r in results]
    fnr_diff = [r['test_measures']['fnr_diff'] for r in results]
    fpr_diff = [r['test_measures']['fpr_diff'] for r in results]
    r_fnr_diff = [r['test_results']['fnr_diff'] for r in results]
    r_fpr_diff = [r['test_results']['fpr_diff'] for r in results]
    subplot.set_autoscaley_on(False)
    subplot.set_ylim([0, 1])
    subplot.set_title(type)
    subplot.plot(weights, acc, 'r-', label="Accuracy", linewidth=3)
    subplot.plot(weights, fnr_diff, 'g-', label="FNR diff", linewidth=3)
    subplot.plot(weights, r_fnr_diff, 'g--', label="relaxed FNR diff", linewidth=3)
    subplot.plot(weights, fpr_diff, 'b-', label="FPR diff", linewidth=3)
    subplot.plot(weights, r_fpr_diff, 'b--', label="relaxed FPR diff", linewidth=3)
    subplot.set_xlabel('Fairness weight')
    subplot.set_ylabel('Rate')
    subplot.legend(loc='best', prop={'size':11}, ncol=1)


def show_results(results_squared, results_abs):
    fig = plt.figure(figsize=(10, 5))
    print('\nThe result for squared relaxation:\n')
    pprint(results_squared)
    sub1 = fig.add_subplot(121)
    plot_results(sub1, results_squared, type='Squared')

    print('\nThe result for absolute value relaxation:\n')
    pprint(results_abs)
    sub2 = fig.add_subplot(122)
    plot_results(sub2, results_abs, type='Absolute value')
    plt.show()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def measure_objective_results(x_test, y_test, protected_index, w):
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
        'fnr_diff': fnr_diff
    }


def logistic(x):
    return np.log(1 + np.exp(x))


def measure_relaxed_results(x_test, y_test, protected_index, w, fp_weight, fn_weight, squared=True):
    x_1, y_1, x_0, y_0 = split_by_protected_value(x_test, y_test, protected_index)
    x_1_pos = get_positive_examples(x_1, y_1)
    x_1_neg = get_negative_examples(x_1, y_1)
    x_0_pos = get_positive_examples(x_0, y_0)
    x_0_neg = get_negative_examples(x_0, y_0)
    diff_pos = (np.sum(x_1_pos, axis=0)/x_1_pos.shape[0]) - (np.sum(x_0_pos, axis=0)/x_0_pos.shape[0])
    diff_neg = (np.sum(x_1_neg, axis=0)/x_1_neg.shape[0]) - (np.sum(x_0_neg, axis=0)/x_0_neg.shape[0])
    log_likelihood = (y_test.T.dot(x_test.dot(w)) - np.sum(logistic(x_test.dot(w))))
    if squared:
        fpr_diff = np.square(diff_neg.dot(w))
        fnr_diff = np.square(diff_pos.dot(w))
    else:
        fpr_diff = np.abs(diff_neg.dot(w))
        fnr_diff = np.abs(diff_pos.dot(w))
    w_norm_square = np.sum(np.square(w))
    return {
        'll': log_likelihood,
        'fnr_diff': fnr_diff,
        'fpr_diff': fpr_diff,
        'objective': -log_likelihood + fp_weight*fpr_diff + fn_weight*fnr_diff
    }


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
        fpr_diff = cp.square(diff_neg*w)
        fnr_diff = cp.square(diff_pos*w)
    else:
        fpr_diff = cp.abs(diff_neg*w)
        fnr_diff = cp.abs(diff_pos*w)

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


def fairness(problem):
    print('\nStart\n')

    x = problem.X
    y = problem.Y
    protected_index = problem.protected_index
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
                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=problem.test_size, random_state=j)

                conv_squared = solve_convex(x_train, y_train, protected_index, gamma, fp_weight=fp_weight, fn_weight=fn_weight, squared=True)
                conv_abs = solve_convex(x_train, y_train, protected_index, gamma, fp_weight=fp_weight, fn_weight=fn_weight, squared=False)

                relaxed_squared = measure_relaxed_results(x_test, y_test, protected_index, conv_squared['w'], fp_weight=fp_weight, fn_weight=fn_weight, squared=True)
                relaxed_abs = measure_relaxed_results(x_test, y_test, protected_index, conv_abs['w'], fp_weight=fp_weight, fn_weight=fn_weight, squared=False)

                measures_squared = measure_objective_results(x_test, y_test, protected_index, conv_squared['w'])
                measures_abs = measure_objective_results(x_test, y_test, protected_index, conv_abs['w'])

                temp_res_squared.append({'train_results': conv_squared, 'test_results': relaxed_squared, 'test_measures': measures_squared})
                temp_res_abs.append({'train_results': conv_abs, 'test_results': relaxed_abs, 'test_measures': measures_abs})

            gamma_result_squared = {'gamma': gamma}
            gamma_result_abs = {'gamma': gamma}
            for key1 in temp_res_squared[0].keys():
                gamma_result_squared[key1] = dict()
                gamma_result_abs[key1] = dict()
                for key2 in temp_res_squared[0][key1].keys():
                    if key2 == 'w':
                        gamma_result_squared[key1][key2] = np.average(np.array([r[key1][key2] for r in temp_res_squared]), axis=0)
                        gamma_result_abs[key1][key2] = np.average(np.array([r[key1][key2] for r in temp_res_abs]), axis=0)
                    else:
                        gamma_result_squared[key1][key2] = np.average(np.array([r[key1][key2] for r in temp_res_squared]))
                        gamma_result_abs[key1][key2] = np.average(np.array([r[key1][key2] for r in temp_res_abs]))

            res_squared.append(gamma_result_squared)
            res_abs.append(gamma_result_abs)

        best_squared = res_squared[np.array([r['test_results']['objective'] for r in res_squared]).argmin()]
        best_abs = res_abs[np.array([r['test_results']['objective'] for r in res_abs]).argmin()]
        best_squared['weight'] = weight
        best_abs['weight'] = weight
        results_squared.append(best_squared)
        results_abs.append(best_abs)
        print(best_squared['gamma'])
        print(best_abs['gamma'])

    print(problem.description)
    show_results(results_squared, results_abs)
    print('\nFinished\n')



