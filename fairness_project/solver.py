import os
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
from sklearn.model_selection import train_test_split
from fairness_data import *


def make_output_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def plot_theta(problem, results, _type):
    x = problem.X
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
        w = results[i]['train_relaxed_measures']['w']
        xp = np.linspace(-1, 2, 100)
        yp = -(w[0, 0]/w[1, 0]*xp) - w[2, 0]/w[1, 0]
        sub.plot(xp, yp, 'k')
        sub.fill_between(xp, yp, 1.5, interpolate=True, color='blue', alpha='0.5')
        sub.fill_between(xp, -0.5, yp, interpolate=True, color='red', alpha='0.5')
        sub.plot(x[:, 0], x[:, 1], 'o', color='black')

    dir_name = os.path.dirname(__file__) + '/../results/' + problem.description
    make_output_dir(dir_name)
    fig.savefig(dir_name + '/' + _type)


def show_theta(problem, results_squared, results_abs):
    plot_theta(problem, results_squared, _type='Squared')
    plot_theta(problem, results_abs, _type='ABS')


def plot_results(subplot, results, _type):
    weights = np.array([r['weight'] for r in results])
    acc = np.array([r['test_real_measures']['accuracy'] for r in results]).reshape(weights.shape)
    fnr_diff = np.array([r['test_real_measures']['fnr_diff'] for r in results]).reshape(weights.shape)
    fpr_diff = np.array([r['test_real_measures']['fpr_diff'] for r in results]).reshape(weights.shape)
    r_fnr_diff = np.array([r['test_relaxed_measures']['fnr_diff'] for r in results]).reshape(weights.shape)
    r_fpr_diff = np.array([r['test_relaxed_measures']['fpr_diff'] for r in results]).reshape(weights.shape)
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


def show_summary(measures_squared, measures_abs, measures_baseline):
    print('\n--------------------------------Runs measures-----------------------------------\n')
    print('\nBaseline measures:\n')
    pprint(measures_baseline)
    print('\nABS measures:\n')
    pprint(measures_abs)
    print('\nSquared measures:\n')
    pprint(measures_squared)
    print('\n---------------------------------Average measures-------------------------------\n')
    for key in measures_squared[0].keys():
        baseline_all = np.array([r[key] for r in measures_baseline])
        abs_all = np.array([r[key] for r in measures_abs])
        squared_all = np.array([r[key] for r in measures_squared])
        print('\nAverage ' + key + ' for Baseline is: ' + str(np.average(baseline_all)))
        print('Average ' + key + ' for ABS is: ' + str(np.average(abs_all)))
        print('Average ' + key + ' for Squared is: ' + str(np.average(squared_all)))


def show_results(results_squared, results_abs, best_for_squared, best_for_abs, problem, run_num):
    fig = plt.figure(figsize=(10, 5))
    fig.subplots_adjust(top=0.9, bottom=0.25, left=0.1, right=0.9, hspace=0.2, wspace=0.25)

    print('----------------------------------------------------------------------------------\n')

    print('\n----------------The result for absolute value relaxation--------------------------\n')
    pprint(results_abs)
    sub1 = fig.add_subplot(121)
    plot_results(sub1, results_abs, _type='Absolute Value')
    print('\n----------------Best Values for Objective absolute value relaxation---------------\n')
    pprint(best_for_abs)

    print('----------------------------------------------------------------------------------\n')

    print('\n----------------The result for squared relaxation---------------------------------\n')
    pprint(results_squared)
    sub2 = fig.add_subplot(122)
    plot_results(sub2, results_squared, _type='Squared')
    print('\n----------------Best Values for Objective squared relaxation----------------------\n')
    pprint(best_for_squared)

    print('----------------------------------------------------------------------------------\n')

    dir_name = os.path.dirname(__file__) + '/../results/' + problem.description
    make_output_dir(dir_name)
    fig.savefig(dir_name + '/' + str(run_num + 1))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def measure_objective_results(x_test, y_test, w,  problem):
    protected_index = problem.protected_index
    objective_weight = problem.objective_weight
    fp = problem.fp
    fn = problem.fn
    y_hat = np.round(sigmoid(np.dot(x_test, w)))
    y_1, y_0 = np.array(split_by_protected_value(x_test, y_test, protected_index))[[1, 3]]
    y_1_hat, y_0_hat = np.array(split_by_protected_value(x_test, y_hat, protected_index))[[1, 3]]

    all_measures = measures(y_test, y_hat)
    _1_measures = measures(y_1, y_1_hat)
    _0_measures = measures(y_0, y_0_hat)
    fpr_diff = abs(_1_measures['fpr'] - _0_measures['fpr'])
    fnr_diff = abs(_1_measures['fnr'] - _0_measures['fnr'])
    return {
        'accuracy': all_measures['acc'],
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


def measure_relaxed_results(x_test, y_test, w, weight, problem, is_squared=True):
    fp_weight = float(weight if problem.fp else 0)
    fn_weight = float(weight if problem.fn else 0)
    protected_index = problem.protected_index

    x_1, y_1, x_0, y_0 = split_by_protected_value(x_test, y_test, protected_index)
    x_1_pos = get_positive_examples(x_1, y_1)
    x_1_neg = get_negative_examples(x_1, y_1)
    x_0_pos = get_positive_examples(x_0, y_0)
    x_0_neg = get_negative_examples(x_0, y_0)
    diff_pos = (np.sum(x_1_pos, axis=0)/x_1_pos.shape[0]) - (np.sum(x_0_pos, axis=0)/x_0_pos.shape[0])
    diff_neg = (np.sum(x_1_neg, axis=0)/x_1_neg.shape[0]) - (np.sum(x_0_neg, axis=0)/x_0_neg.shape[0])
    log_likelihood = (y_test.T.dot(x_test.dot(w)) - np.sum(logistic(x_test.dot(w))))
    if is_squared:
        fpr_diff = np.square(diff_neg.dot(w))
        fnr_diff = np.square(diff_pos.dot(w))
    else:
        fpr_diff = np.abs(diff_neg.dot(w))
        fnr_diff = np.abs(diff_pos.dot(w))
    return {
        'll': log_likelihood.item(),
        'fnr_diff': fnr_diff.item(),
        'fpr_diff': fpr_diff.item(),
        'objective': (-log_likelihood + fp_weight*fpr_diff + fn_weight*fnr_diff).item()
    }


def solve_one_time_by_type(problem, x_train, y_train, x_test, y_test, gamma, weight, is_squared=True):
    fp_weight = float(weight if problem.fp else 0)
    fn_weight = float(weight if problem.fn else 0)
    protected_index = problem.protected_index

    x_1, y_1, x_0, y_0 = split_by_protected_value(x_train, y_train, protected_index)
    x_1_pos = get_positive_examples(x_1, y_1)
    x_1_neg = get_negative_examples(x_1, y_1)
    x_0_pos = get_positive_examples(x_0, y_0)
    x_0_neg = get_negative_examples(x_0, y_0)

    w = cp.Variable(x_train.shape[1])

    diff_pos = (np.sum(x_1_pos, axis=0)/x_1_pos.shape[0]) - (np.sum(x_0_pos, axis=0)/x_0_pos.shape[0])
    diff_neg = (np.sum(x_1_neg, axis=0)/x_1_neg.shape[0]) - (np.sum(x_0_neg, axis=0)/x_0_neg.shape[0])
    log_likelihood = (y_train.T * (x_train * w) - cp.sum(cp.logistic(x_train * w)))
    if is_squared:
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
    p.solve(solver='ECOS', verbose=False, max_iters=1000)
    solution = {
        'w': w.value,
        'll': log_likelihood.value,
        'fnr_diff': fnr_diff.value,
        'fpr_diff': fpr_diff.value,
        'objective': -log_likelihood.value + fn_weight*fnr_diff.value + fp_weight*fpr_diff.value
    }
    train_real_measures = measure_objective_results(x_train, y_train, solution['w'], problem)
    test_relaxed_measures = measure_relaxed_results(x_test, y_test, solution['w'], weight, problem, is_squared=is_squared)
    test_real_measures = measure_objective_results(x_test, y_test, solution['w'], problem)

    return {
        'weight': weight,
        'gamma': gamma,
        'train_relaxed_measures': solution,
        'train_real_measures': train_real_measures,
        'test_relaxed_measures': test_relaxed_measures,
        'test_real_measures': test_real_measures
    }


def solve_convex(problem, run_num):
    x = problem.X
    y = problem.Y
    gammas = np.linspace(problem.gamma_gt, problem.gamma_lt, num=problem.gamma_res)
    weights = np.linspace(problem.weight_gt, problem.weight_lt, num=problem.weight_res)

    x_train_all, x_test, y_train_all, y_test = train_test_split(x, y, test_size=problem.test_size, random_state=run_num)
    results_squared = list()
    results_abs = list()
    for weight in weights:
        temp_results_squared = dict()
        temp_results_abs = dict()
        for fold in range(problem.num_of_folds):
            x_train, x_val, y_train, y_val = train_test_split(x_train_all, y_train_all, test_size=problem.val_size, random_state=fold)
            for gamma_index in range(problem.gamma_res):
                if gamma_index not in temp_results_squared:
                    temp_results_squared[gamma_index] = list()
                if gamma_index not in temp_results_abs:
                    temp_results_abs[gamma_index] = list()
                gamma = gammas[gamma_index]
                try:
                    solution =\
                        solve_one_time_by_type(problem, x_train, y_train, x_val, y_val, gamma, weight, is_squared=True)
                    temp_results_squared[gamma_index].append(
                        solution['test_real_measures'] if (weight >= 0.001) else solution['test_relaxed_measures']
                    )
                except Exception as e:
                    print(str(e))
                    print('Squared:\tFailed for gamma: ' + str(gamma) + ", weight: " + str(weight) + '\n')
                    pass
                try:
                    solution =\
                        solve_one_time_by_type(problem, x_train, y_train, x_val, y_val, gamma, weight, is_squared=False)
                    temp_results_abs[gamma_index].append(
                        solution['test_real_measures'] if (weight >= 0.001) else solution['test_relaxed_measures']
                    )
                except Exception as e:
                    print(str(e))
                    print('ABS:\t\tFailed for gamma: ' + str(gamma) + ", weight: " + str(weight) + '\n')
                    pass

        avg_results_squared = list()
        avg_results_abs = list()
        for gamma_index in range(problem.gamma_res):
            gamma = gammas[gamma_index]
            avg_squared = {'gamma': gamma}
            avg_abs = {'gamma': gamma}
            for key in temp_results_squared[0][0].keys():
                avg_squared[key] = np.average(np.array([r[key] for r in temp_results_squared[gamma_index]]))
                avg_abs[key] = np.average(np.array([r[key] for r in temp_results_abs[gamma_index]]))
            avg_results_squared.append(avg_squared)
            avg_results_abs.append(avg_abs)

        argmin_gamma_squared = avg_results_squared[np.array([r['objective'] for r in avg_results_squared]).argmin()]['gamma']
        argmin_gamma_abs = avg_results_abs[np.array([r['objective'] for r in avg_results_abs]).argmin()]['gamma']
        print('Squared:\tthe best gamma for weight: ' + str(weight) + ' is: ' + str(argmin_gamma_squared))
        print('ABS:\t\tthe best gamma for weight: ' + str(weight) + ' is: ' + str(argmin_gamma_abs) + '\n')

        try:
            solution_squared =\
                solve_one_time_by_type(problem, x_train_all, y_train_all, x_test, y_test, argmin_gamma_squared, weight, is_squared=True)
            results_squared.append(solution_squared)
        except Exception as e:
            print(str(e))
            print('Squared:\tFailed for weight: ' + str(weight) + '\n')
            pass
        try:
            solution_abs =\
                solve_one_time_by_type(problem, x_train_all, y_train_all, x_test, y_test, argmin_gamma_abs, weight, is_squared=False)
            results_abs.append(solution_abs)
        except Exception as e:
            print(str(e))
            print('ABS:\t\tFailed for weight: ' + str(weight) + '\n')
            pass

    return results_squared, results_abs


def solve_problem_for_fairness(problem, synthetic=False):
    print('\n----------------------------------START--------------------------------------\n')

    measures_squared = list()
    measures_abs = list()
    measures_baseline = list()
    for run in range(problem.num_of_runs):
        results_squared, results_abs = solve_convex(problem=problem, run_num=run)
        results_baseline = results_abs[0]

        best_train_squared = results_squared[np.array([r['train_real_measures']['objective'] for r in results_squared]).argmin()]
        best_train_abs = results_abs[np.array([r['train_real_measures']['objective'] for r in results_abs]).argmin()]

        measures_squared.append({
            'accuracy': best_train_squared['test_real_measures']['accuracy'],
            'fpr_diff': best_train_squared['test_real_measures']['fpr_diff'],
            'fnr_diff': best_train_squared['test_real_measures']['fnr_diff']
        })
        measures_abs.append({
            'accuracy': best_train_abs['test_real_measures']['accuracy'],
            'fpr_diff': best_train_abs['test_real_measures']['fpr_diff'],
            'fnr_diff': best_train_abs['test_real_measures']['fnr_diff']
        })
        measures_baseline.append({
            'accuracy': results_baseline['test_real_measures']['accuracy'],
            'fpr_diff': results_baseline['test_real_measures']['fpr_diff'],
            'fnr_diff': results_baseline['test_real_measures']['fnr_diff']})

        show_results(results_squared, results_abs, best_train_squared, best_train_abs, problem, run)
        if synthetic:
            show_theta(problem, results_squared, results_abs)

    show_summary(measures_squared, measures_abs, measures_baseline)
    print('\n----------------------------------DONE--------------------------------------\n')
    # plt.show()

