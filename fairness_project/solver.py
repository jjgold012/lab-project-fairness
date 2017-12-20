import cvxpy as cp
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from fairness_project.fairness_data import *


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def solve_non_convex():
    print('non-convex')


def solve_convex(x, y, fp_weight, fn_weight, gamma, protected_index, squared=True):
    x_1, y_1, x_0, y_0 = split_by_protected_value(x, y, protected_index)
    x_1_pos = get_positive_examples(x_1, y_1)
    x_1_neg = get_negative_examples(x_1, y_1)
    x_0_pos = get_positive_examples(x_0, y_0)
    x_0_neg = get_negative_examples(x_0, y_0)
    w = cp.Variable(x.shape[1])

    log_likelihood = (y.T*(x*w) - cp.sum_entries(cp.logistic(x*w)))
    
    if squared:
        fpr = \
            cp.square(((np.sum(x_0_neg, axis=0)/x_0_neg.shape[0]) - (np.sum(x_1_neg, axis=0)/x_1_neg.shape[0]))*w)
        fnr = \
            cp.square(((np.sum(x_1_pos, axis=0)/x_1_pos.shape[0]) - (np.sum(x_0_pos, axis=0)/x_0_pos.shape[0]))*w)
    else:
        fpr = \
            cp.abs(((np.sum(x_0_neg, axis=0)/x_0_neg.shape[0]) - (np.sum(x_1_neg, axis=0)/x_1_neg.shape[0]))*w)
        fnr = \
            cp.abs(((np.sum(x_1_pos, axis=0)/x_1_pos.shape[0]) - (np.sum(x_0_pos, axis=0)/x_0_pos.shape[0]))*w)

    
    # reg_fpr_orig = \
    #     problem.fp_fn_weight * cp.abs((cp.sum_entries(cp.logistic(problem.X_0_neg * cp.neg(w))) / float(problem.X_0_neg.shape[0])) - (cp.sum_entries(cp.logistic(problem.X_1_neg * cp.neg(w))) / float(problem.X_1_neg.shape[0])))
    #
    # reg_fnr_orig = \
    #     problem.fp_fn_weight * cp.abs((cp.sum_entries(cp.logistic(problem.X_1_pos * cp.neg(w))) / float(problem.X_1_pos.shape[0])) - (cp.sum_entries(cp.logistic(problem.X_0_pos * cp.neg(w))) / float(problem.X_0_pos.shape[0])))

    w_norm_square = cp.sum_squares(w)

    objective = cp.Minimize(-log_likelihood +
                            fn_weight*fnr +
                            fp_weight*fpr +
                            gamma*w_norm_square)

    p = cp.Problem(objective)
    p.solve()
    return {'w': w.value, 'll': log_likelihood.value, 'fnr': fnr.value, 'fpr': fpr.value}


def fairness(problem: FairnessProblem):
    x = problem.X
    y = problem.Y

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)
    gamma = 0.5
    fp_fn_weight = problem.fp_fn_weight
    res = solve_convex(x_train, y_train, fp_fn_weight, fp_fn_weight, gamma, problem.protected_index)
    w = res['w']
    print(w)

    y_hat = np.round(sigmoid(np.dot(x_test, w)))
    measures(y_test, y_hat)
    equalized_odds_reg(x_train,y_train,y_train,problem.protected_index)



# def original(problem):
#
#     # w=cvx.Variable(3,1)
#     # obj=cvx.Maximize(y.T*X*w-cvx.sum_entries(cvx.logistic(X*w)))
#     # prob=cvx.Problem(obj).solve()
#     # w=w.value
#     # xp=np.linspace(0,2,100).reshape(-1,1)
#     # yp=-w[1,0]/w[2,0]*xp-w[0,0]/w[2,0]
#     # plt.figure(figsize=(10,6))
#     # plt.plot(X[C1,1],X[C1,2],'ro',label='C1')
#     # plt.plot(X[C2,1],X[C2,2],'bo',label='C2')
#     # plt.plot(xp,yp,'k',label='Logistic Regression')
#     # plt.legend()
#     # plt.show()
#
#     print("orig")

