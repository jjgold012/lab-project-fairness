import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
from sklearn.model_selection import train_test_split
from fairness_data import *


def plot_theta(x, results, _type):
    num_of_results = min([len(results), 4])
    fig = plt.figure(figsize=(3.6*num_of_results, 3.5))
    fig.subplots_adjust(wspace=0.5, bottom=0.25, left=0.1, right=0.95)
    fig.canvas.set_window_title(_type)
    results = [results[int(round(i))] for i in np.linspace(0, len(results) - 1, num_of_results)]
    for i in range(num_of_results):
        sub = fig.add_subplot(1, num_of_results, i + 1)
        sub.set_ylim([-0.5, 1.5])
        sub.set_xlim([-0.5, 1.5])
        sub.set_title(str(results[i]['weight']))
        sub.set_xlabel('A', fontweight='bold')
        sub.set_ylabel('X', fontweight='bold')
        w = results[i]['train_results']['w']
        xp = np.linspace(-1, 2, 100)
        yp = -(w[0, 0]/w[1, 0]*xp) - w[2, 0]/w[1, 0]
        sub.plot(xp, yp, 'k')
        sub.fill_between(xp, yp, 1.5, interpolate=True, color='blue', alpha='0.5')
        sub.fill_between(xp, -0.5, yp, interpolate=True, color='red', alpha='0.5')
        sub.plot(x[:, 0], x[:, 1], 'o', color='black')


def show_theta(x, results_squared, results_abs):
    plot_theta(x, results_squared, _type='Squared')
    plot_theta(x, results_abs, _type='ABS')


def plot_results(subplot, results, _type):
    weights = np.array([r['weight'] for r in results])
    acc = np.array([r['test_measures']['acc'] for r in results]).reshape(weights.shape)
    fnr_diff = np.array([r['test_measures']['fnr_diff'] for r in results]).reshape(weights.shape)
    fpr_diff = np.array([r['test_measures']['fpr_diff'] for r in results]).reshape(weights.shape)
    r_fnr_diff = np.array([r['test_results']['fnr_diff'] for r in results]).reshape(weights.shape)
    r_fpr_diff = np.array([r['test_results']['fpr_diff'] for r in results]).reshape(weights.shape)
    subplot.set_autoscaley_on(False)
    subplot.set_ylim([0, 1])
    subplot.plot(weights, acc, 'r-', label="Accuracy", linewidth=2)
    subplot.plot(weights, fpr_diff, 'b-', label="FPR Difference", linewidth=2)
    subplot.plot(weights, r_fpr_diff, 'b--', label="Relaxed FPR Diff.", linewidth=2)
    subplot.plot(weights, fnr_diff, 'g-', label="FNR Difference", linewidth=2)
    subplot.plot(weights, r_fnr_diff, 'g--', label="Relaxed FNR Diff.", linewidth=2)
    subplot.set_title(_type)
    subplot.set_xlabel('Fairness Penalizers Weight')
    subplot.set_ylabel('Rate')
    subplot.legend(loc='best', prop={'size':11}, ncol=1)


def show_results(results_squared, results_abs):
    fig = plt.figure(figsize=(10, 5))
    print('\n----------------The result for absolute value relaxation--------------------------\n')
    pprint(results_abs)
    sub1 = fig.add_subplot(121)
    plot_results(sub1, results_abs, _type='Absolute Value')
    print('\n----------------Best Values for Objective absolute value relaxation---------------\n')
    best_abs = results_abs[np.array([r['train_measures']['objective'] for r in results_abs]).argmin()]
    pprint(best_abs)
    print('----------------------------------------------------------------------------------\n')

    print('\n----------------The result for squared relaxation---------------------------------\n')
    pprint(results_squared)
    sub2 = fig.add_subplot(122)
    plot_results(sub2, results_squared, _type='Squared')
    print('\n----------------Best Values for Objective squared relaxation----------------------\n')
    best_squared = results_squared[np.array([r['train_measures']['objective'] for r in results_squared]).argmin()]
    pprint(best_squared)
    print('----------------------------------------------------------------------------------\n')


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def measure_objective_results(x_test, y_test, protected_index, fp, fn, objective_weight, w):
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
        'objective': (1 - all_measures['acc']) + (objective_weight*fpr_diff if fp else 0) + (objective_weight*fnr_diff if fn else 0)
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
    p.solve(solver='ECOS', verbose=False, max_iters=400)

    return {
        'w': w.value,
        'll': log_likelihood.value,
        'fnr_diff': fnr_diff.value,
        'fpr_diff': fpr_diff.value,
        'objective': -log_likelihood.value + fn_weight*fnr_diff.value + fp_weight*fpr_diff.value
    }


def fairness(problem, synthetic=False):
    print('\nStart\n')

    x = problem.X
    y = problem.Y
    x_train_all, x_test, y_train_all, y_test = train_test_split(x, y, test_size=problem.test_size, random_state=1)
    protected_index = problem.protected_index
    results_squared = list()
    results_abs = list()
    for weight in np.linspace(problem.weight_gt, problem.weight_lt, num=problem.weight_res):
        res_squared = list()
        res_abs = list()
        fp_weight = float(weight if problem.fp else 0)
        fn_weight = float(weight if problem.fn else 0)
        for gamma in np.linspace(problem.gamma_gt, problem.gamma_lt, num=problem.gamma_res):
            temp_res_squared = list()
            temp_res_abs = list()
            for j in range(problem.num_of_tries):
                x_train, x_val, y_train, y_val = train_test_split(x_train_all, y_train_all, test_size=problem.val_size, random_state=j)
                try:
                    conv_squared = solve_convex(x_train, y_train, protected_index, gamma, fp_weight=fp_weight, fn_weight=fn_weight, squared=True)
                    measures_train_squared = measure_objective_results(x_train, y_train, protected_index, problem.fp, problem.fn, problem.objective_weight, conv_squared['w'])
                    relaxed_val_squared = measure_relaxed_results(x_val, y_val, protected_index, conv_squared['w'], fp_weight=fp_weight, fn_weight=fn_weight, squared=True)
                    measures_val_squared = measure_objective_results(x_val, y_val, protected_index, problem.fp, problem.fn, problem.objective_weight, conv_squared['w'])
                    temp_res_squared.append({'train_results': conv_squared, 'train_measures': measures_train_squared, 'val_results': relaxed_val_squared, 'val_measures': measures_val_squared})
                except:
                    pass
                try:
                    conv_abs = solve_convex(x_train, y_train, protected_index, gamma, fp_weight=fp_weight, fn_weight=fn_weight, squared=False)
                    measures_train_abs = measure_objective_results(x_train, y_train, protected_index, problem.fp, problem.fn, problem.objective_weight, conv_abs['w'])
                    relaxed_val_abs = measure_relaxed_results(x_val, y_val, protected_index, conv_abs['w'], fp_weight=fp_weight, fn_weight=fn_weight, squared=False)
                    measures_val_abs = measure_objective_results(x_val, y_val, protected_index, problem.fp, problem.fn, problem.objective_weight, conv_abs['w'])
                    temp_res_abs.append({'train_results': conv_abs, 'train_measures': measures_train_abs, 'val_results': relaxed_val_abs, 'val_measures': measures_val_abs})
                except:
                    pass
            print("Squared -\tsuccess: " + str(len(temp_res_squared)) + "\t\tweight: " + str(weight) + "\t\tgamma: " + str(gamma))
            print("ABS -\t\tsuccess: " + str(len(temp_res_abs)) + "\t\tweight: " + str(weight) + "\t\tgamma: " + str(gamma))
            gamma_result_squared = {'gamma': gamma}
            gamma_result_abs = {'gamma': gamma}
            for key1 in temp_res_squared[0].keys():
                gamma_result_squared[key1] = dict()
                gamma_result_abs[key1] = dict()
                for key2 in temp_res_squared[0][key1].keys():
                    squared_all = np.array([r[key1][key2] for r in temp_res_squared])
                    abs_all = np.array([r[key1][key2] for r in temp_res_abs])
                    if key2 == 'w':
                        gamma_result_squared[key1][key2] = np.average(squared_all, axis=0)
                        gamma_result_abs[key1][key2] = np.average(abs_all, axis=0)
                    else:
                        gamma_result_squared[key1][key2] = np.average(squared_all)
                        gamma_result_abs[key1][key2] = np.average(abs_all)

            res_squared.append(gamma_result_squared)
            res_abs.append(gamma_result_abs)

        best_squared = {'weight': weight, 'gamma': res_squared[np.array([r['val_measures']['objective'] for r in res_squared]).argmin()]['gamma']}

        conv_squared = solve_convex(x_train_all, y_train_all, protected_index, best_squared['gamma'], fp_weight=fp_weight, fn_weight=fn_weight, squared=True)
        measures_train_squared = measure_objective_results(x_train_all, y_train_all, protected_index, problem.fp, problem.fn, problem.objective_weight, conv_squared['w'])
        relaxed_test_squared = measure_relaxed_results(x_test, y_test, protected_index, conv_squared['w'], fp_weight=fp_weight, fn_weight=fn_weight, squared=True)
        measures_test_squared = measure_objective_results(x_test, y_test, protected_index, problem.fp, problem.fn, problem.objective_weight, conv_squared['w'])
        best_squared['train_results'] = conv_squared
        best_squared['train_measures'] = measures_train_squared
        best_squared['test_results'] = relaxed_test_squared
        best_squared['test_measures'] = measures_test_squared

        best_abs = {'weight': weight, 'gamma': res_abs[np.array([r['val_measures']['objective'] for r in res_abs]).argmin()]['gamma']}

        conv_abs = solve_convex(x_train_all, y_train_all, protected_index, best_abs['gamma'], fp_weight=fp_weight, fn_weight=fn_weight, squared=False)
        measures_train_abs = measure_objective_results(x_train_all, y_train_all, protected_index, problem.fp, problem.fn, problem.objective_weight, conv_abs['w'])
        relaxed_test_abs = measure_relaxed_results(x_test, y_test, protected_index, conv_abs['w'], fp_weight=fp_weight, fn_weight=fn_weight, squared=False)
        measures_test_abs = measure_objective_results(x_test, y_test, protected_index, problem.fp, problem.fn, problem.objective_weight, conv_abs['w'])
        best_abs['train_results'] = conv_abs
        best_abs['train_measures'] = measures_train_abs
        best_abs['test_results'] = relaxed_test_abs
        best_abs['test_measures'] = measures_test_abs

        results_squared.append(best_squared)
        results_abs.append(best_abs)
        print("\nSquared best gamma: " + str(best_squared['gamma']) + "\n")
        print("\nABS best gamma: " + str(best_abs['gamma']) + "\n")

    print(problem.original_options)
    show_results(results_squared, results_abs)
    if synthetic:
        show_theta(x, results_squared, results_abs)
    print('Done')
    plt.show()



