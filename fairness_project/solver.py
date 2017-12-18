import cvxpy as cvx
import numpy as np
import matplotlib.pyplot as plt


def fairness(fairness_problem):
    print("fairness")



def original(problem):

    # w=cvx.Variable(3,1)
    # obj=cvx.Maximize(y.T*X*w-cvx.sum_entries(cvx.logistic(X*w)))
    # prob=cvx.Problem(obj).solve()
    # w=w.value
    # xp=np.linspace(0,2,100).reshape(-1,1)
    # yp=-w[1,0]/w[2,0]*xp-w[0,0]/w[2,0]
    # plt.figure(figsize=(10,6))
    # plt.plot(X[C1,1],X[C1,2],'ro',label='C1')
    # plt.plot(X[C2,1],X[C2,2],'bo',label='C2')
    # plt.plot(xp,yp,'k',label='Logistic Regression')
    # plt.legend()
    # plt.show()

    print("orig")

