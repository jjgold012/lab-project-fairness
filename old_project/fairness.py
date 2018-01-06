from sys import stdout
from csv import DictReader, DictWriter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
# from sklearn import preprocessing
# from sklearn.cross_validation import cross_val_score
# from pandas.tools.plotting import scatter_matrix
# from pandas.tools.plotting import parallel_coordinates
# from IPython.external.decorators import _numpy_testing_utils


class PeekyReader:
    def __init__(self, reader):
        self.peeked = None
        self.reader = reader

    def peek(self):
        if self.peeked is None:
            self.peeked = next(self.reader)
        return self.peeked

    def __iter__(self):
        return self

    def __next__(self):
        if self.peeked is not None:
            ret = self.peeked
            self.peeked = None
            return ret
        try:
            return next(self.reader)
        except StopIteration:
            self.peeked = None
            raise StopIteration


class Person:
    def __init__(self, reader):
        self.__rows = []
        self.__idx = reader.peek()['id']
        try:
            while reader.peek()['id'] == self.__idx:
                self.__rows.append(next(reader))
        except StopIteration:
            pass

    @property
    def lifetime(self):
        memo = 0
        for it in self.__rows:
            memo += int(it['end']) - int(it['start'])
        return memo

    @property
    def age(self):
        return int(self.__rows[0]['age'])
    
    @property
    def age_category(self):
        if self.age < 25:
            return 1
        elif self.age <= 45:
            return 2
        else:
            return 3

    @property
    def days_b_screening_arrest(self):
        if self.__rows[0]['days_b_screening_arrest'] == '':
            return np.inf
        return int(self.__rows[0]['days_b_screening_arrest'])

    
    @property
    def recidivist(self):
        return (self.__rows[0]['two_year_recid'] == "1")

    @property
    def recidivism(self):
        return int(self.__rows[0]['is_recid'])
    
    @property
    def c_charge_degree(self):
        return self.__rows[0]['c_charge_degree']
    
    @property
    def score_text(self):
        return self.__rows[0]['score_text']
    
    @property
    def violent_recidivist(self):
        return (self.__rows[0]['is_violent_recid'] == "1" and
                self.lifetime <= 730)

    @property
    def gender(self):
        return self.__rows[0]['sex']
    
    @property
    def gender_num(self):
        if self.__rows[0]['sex'] == 'male':
            return 1
        else:
            return 2
    
    @property
    def priors(self):
        return int(self.__rows[0]['priors_count'])
    
    @property
    def c_charge_degree(self):
        return self.__rows[0]['c_charge_degree']

    @property
    def low(self):
        return self.__rows[0]['score_text'] == "Low"

    @property
    def high(self):
        return not self.low

    @property
    def low_med(self):
        return self.low or self.score == "Medium"

    @property
    def true_high(self):
        return self.score == "High"

    @property
    def vlow(self):
        return self.__rows[0]['v_score_text'] == "Low"

    @property
    def vhigh(self):
        return not self.vlow

    @property
    def vlow_med(self):
        return self.vlow or self.vscore == "Medium"

    @property
    def vtrue_high(self):
        return self.vscore == "High"

    @property
    def score(self):
        return self.__rows[0]['score_text']

    @property
    def vscore(self):
        return self.__rows[0]['v_score_text']

    @property
    def race(self):
        return self.__rows[0]['race']
    
    @property
    def race_num(self):
        if self.__rows[0]['race'] == 'Caucasian':
            return 1
        else:
            return 2
    
    @property
    def risk_score(self):
        return int(self.__rows[0]['decile_score'])

    @property
    def valid(self):
        return (self.__rows[0]['is_recid'] != "-1" and
                (self.recidivist and self.lifetime <= 730) or
                self.lifetime > 730)

    @property
    def compas_felony(self):
        return 'F' in self.__rows[0]['c_charge_degree']

    @property
    def charge_degree_num(self):
        if 'M' in self.__rows[0]['c_charge_degree']:
            return 1
        else:
            return 2

    @property
    def score_valid(self):
        return self.score in ["Low", "Medium", "High"]

    @property
    def vscore_valid(self):
        return self.vscore in ["Low", "Medium", "High"]

    @property
    def rows(self):
        return self.__rows


def count(fn, data):
    return len(list(filter(fn, list(data))))


def t(tn, fp, fn, tp):
    surv = tn + fp
    recid = tp + fn
    print("           \tLow\tHigh")
    print("Survived   \t%i\t%i\t%.2f" % (tn, fp, surv / (surv + recid)))
    print("Recidivated\t%i\t%i\t%.2f" % (fn, tp, recid / (surv + recid)))
    print("Total: %.2f" % (surv + recid))
    print("False positive rate: %.2f" % (fp / surv * 100))
    print("False negative rate: %.2f" % (fn / recid * 100))
    spec = tn / (tn + fp)
    sens = tp / (tp + fn)
    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)
    prev = recid / (surv + recid)
    print("Specificity: %.2f" % spec)
    print("Sensitivity: %.2f" % sens)
    print("Prevalence: %.2f" % prev)
    print("PPV: %.2f" % ppv)
    print("NPV: %.2f" % npv)
    print("LR+: %.2f" % (sens / (1 - spec)))
    print("LR-: %.2f" % ((1-sens) / spec))


def table(recid, surv, prefix=''):
    tn = count(lambda i: getattr(i, prefix + 'low'), surv)
    fp = count(lambda i: getattr(i, prefix + 'high'), surv)
    fn = count(lambda i: getattr(i, prefix + 'low'), recid)
    tp = count(lambda i: getattr(i, prefix + 'high'), recid)
    t(tn, fp, fn, tp)


def hightable(recid, surv, prefix=''):
    tn = count(lambda i: getattr(i, prefix + 'low_med'), surv)
    fp = count(lambda i: getattr(i, prefix + 'true_high'), surv)
    fn = count(lambda i: getattr(i, prefix + 'low_med'), recid)
    tp = count(lambda i: getattr(i, prefix + 'true_high'), recid)
    t(tn, fp, fn, tp)


def vtable(recid, surv):
    table(recid, surv, prefix='v')


def vhightable(recid, surv):
    hightable(recid, surv, prefix='v')


def is_race(race):
    return lambda x: x.race == race


def create_two_year_files():
    people = []
    headers = []
    with open("./compas-scores-two-years.csv") as f:
        reader = PeekyReader(DictReader(f))
        try:
            while True:
                p = Person(reader)
                people.append(p)
        except StopIteration:
            pass
        headers = reader.reader.fieldnames
        
    pop = list(filter(lambda q: q.race=="African-American" or q.race=="Caucasian", 
                     filter(lambda w: (w.days_b_screening_arrest <= 30 and w.days_b_screening_arrest >= -30), 
                             filter(lambda e: e.recidivism != -1, 
                                    filter(lambda r: r.c_charge_degree != 'O', 
                                         filter(lambda t: t.score_text != 'N/A', people))))))
    return pop


def set_data(data_list):
    m = len(data_list)
    n = 5
    
    X = np.zeros((m,n))
    y = np.zeros(m)
    i = -1;
    
    for person in data_list:
        i = i+1
        X[i,:] = [person.age_category, person.gender_num, person.race_num, person.priors, person.charge_degree_num]
        y[i] = person.recidivist
    return X,y
    
    
    
    
def perf_measure(y_actual, y_hat):

    POS = 0
    NEG = 0
    
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
           POS += 1
           TP += 1
    for i in range(len(y_hat)):
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           POS += 1
           FP += 1
    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==0:
           NEG += 1
           TN += 1
    for i in range(len(y_hat)): 
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           NEG += 1
           FN += 1

    #print(TP)
    #print(TN)
    #print(FP)
    #print(FN)
    return [float(TP)/(TP+FN), float(FP)/(FP+TN), float(TN)/(TN+FP), float(FN)/(FN+TP)]    
    
  
    ############ Logistic Regression ###########
    
    
def sigmoid(scores):
    return 1 / (1 + np.exp(-scores))
    
def log_likelihood(features, target, weights):
    scores = np.dot(features, weights)
    ll = np.sum( target*scores - np.log(1 + np.exp(scores)) )
    return ll


def logistic_regression(X, Y, X_test, Y_test, num_steps, learning_rate, add_intercept = False):
    
    perf_train = np.zeros((num_steps,6))
    perf_test = np.zeros((num_steps,6))
    
    if add_intercept:
        intercept = np.ones((X.shape[0], 1))
        X = np.hstack((intercept, X))
        
    weights = np.zeros(X.shape[1])
    weights2 = np.zeros(X.shape[1])
    
    FPR_regularizer_rate = 0
    FNR_regularizer_rate = 0.5
    l2_regularizer_rate_case1 = 0.001
    l2_regularizer_rate_case2 = 0.007
    
    
    for step in range(num_steps):
        #FPR_regularizer_rate *= 0.999
        #FNR_regularizer_rate *= 0.999
        
        scores = np.dot(X, weights)
        Y_hat = sigmoid(scores)
        acc_train = 1 - (np.sum(np.abs(Y-np.round(Y_hat)))/X.shape[0])
        
        scores = np.dot(X_test, weights)
        Y_hat_test = sigmoid(scores)
        acc_test = 1 - (np.sum(np.abs(Y_test-np.round(Y_hat_test)))/X_test.shape[0])
        
        scores2 = np.dot(X, weights2)
        Y_hat2 = sigmoid(scores2)
        acc2_train = 1 - (np.sum(np.abs(Y-np.round(Y_hat2)))/X.shape[0])
        
        scores2 = np.dot(X_test, weights2)
        Y_hat2_test = sigmoid(scores2)
        acc2_test = 1 - (np.sum(np.abs(Y_test-np.round(Y_hat2_test)))/X_test.shape[0]) 
        
        #print(Y_test-np.round(Y_hat2))
        
        perf_train[step,0] = acc_train
        perf_test[step,0] = acc_test
        perf_train[step,3] = acc2_train
        perf_test[step,3] = acc2_test
     
        #print((acc_train,acc2_train,acc_test,acc2_test))
        #print(acc2_test)   
        #if step % 1 == 0 and step != 1:
            #plt.subplot(2, 1, 1)
            #plt.plot(step, np.sum(1-(np.abs(Y-np.round(Y_hat))))/X.shape[0], 'go')
            #plt.subplot(2, 1, 2)
            #plt.plot(step, np.sum(1-(Y-np.round(Y_hat2)))/X.shape[0], 'go')
            
            
        # Update weights with gradient
        output_error_signal = Y - Y_hat
        gradient = np.dot(X.T, output_error_signal)
        output_error_signal2 = Y - Y_hat2
        gradient2 = np.dot(X.T, output_error_signal2)
        [FPR_regularizer_gradient,FNR_regularizer_gradient,FPR_difference,FNR_difference,FPR_difference_temp,FNR_difference_temp] = equalized_odds_regularizer(X, y, Y_hat)
        [FPR_regularizer_gradient2,FNR_regularizer_gradient2,FPR_difference2,FNR_difference2,FPR_difference_temp2,FNR_difference_temp2] = equalized_odds_regularizer(X, y, Y_hat2)
        
    
        
        
        perf_train[step,1] = -FPR_difference_temp
        perf_train[step,2] = -FNR_difference_temp
        perf_train[step,4] = -FPR_difference_temp2
        perf_train[step,5] = -FNR_difference_temp2
        
        #print(FPR_difference,FNR_difference)
        #print(FPR_difference,FNR_difference,FPR_difference_temp,FNR_difference_temp)
        
        [t1,t2,FPR_difference_test,FNR_difference_test,FPR_difference_test_temp,FNR_difference_test_temp] = equalized_odds_regularizer(X_test, y_test, Y_hat_test)
        [t3,t4,FPR_difference2_test,FNR_difference2_test,FPR_difference_test_temp2,FNR_difference_test_temp2] = equalized_odds_regularizer(X_test, y_test, Y_hat2_test)
        
        perf_test[step,1] = -FPR_difference_test_temp
        perf_test[step,2] = -FNR_difference_test_temp
        perf_test[step,4] = -FPR_difference_test_temp2
        perf_test[step,5] = -FNR_difference_test_temp2
        
        #if step > 5:
            #if np.abs(FPR_difference_test_temp)<0.1 and np.abs(FNR_difference_test_temp)<0.1:
                #FPR_regularizer_rate *= 0.1
                #FNR_regularizer_rate *= 0.1
        
        
        #print(step)
        #n = preprocessing.normalize(regularizer_gradient,'l2')
        if step != 199:
            weights += learning_rate * gradient - (l2_regularizer_rate_case1*2*weights) + (FPR_regularizer_rate * FPR_regularizer_gradient) + (FNR_regularizer_rate * FNR_regularizer_gradient)
            weights2 += learning_rate * gradient2 - (l2_regularizer_rate_case2*2*weights2)
         
        
        # Print log-likelihood every so often
        #if step % 50 == 0:
            # red dashes, blue squares and green triangles
            print(log_likelihood(X, np.round(Y_hat), weights))
        #print(weights)
    #plt.show()
    #print(perf_train)
    #print(perf_train) 
    return weights,perf_train,perf_test
    
    
def equalized_odds_regularizer(X,Y,Y_hat):
   
    
    # White
    X_white = X[np.where(X[:,2] == 1)]
    Y_white = Y[np.where(X[:,2] == 1)]
    Y_hat_white = Y_hat[np.where(X[:,2] == 1)]
    
    # Black
    X_black = X[np.where(X[:,2] == 2)]
    Y_black = Y[np.where(X[:,2] == 2)]
    Y_hat_black = Y_hat[np.where(X[:,2] == 2)]
    
    # White, Positive
    X_white_pos = X_white[np.where(Y_white == 1)]
    Y_white_pos = Y_white[np.where(Y_white == 1)]
    Y_hat_white_pos = Y_hat_white[np.where(Y_white == 1)]
    N_white_pos = X_white_pos.shape[0]
    Y_hat_white_pos_sum = np.sum(Y_hat_white_pos)
    X_white_pos_sum = np.sum(X_white_pos, axis=0)
    
    # White, Negative
    X_white_neg = X_white[np.where(Y_white == 0)]
    Y_white_neg = Y_white[np.where(Y_white == 0)]
    Y_hat_white_neg = Y_hat_white[np.where(Y_white == 0)]
    N_white_neg = X_white_neg.shape[0]
    Y_hat_white_neg_sum = np.sum(Y_hat_white_neg)
    X_white_neg_sum = np.sum(X_white_neg, axis=0)
    
    # Black, Positive
    X_black_pos = X_black[np.where(Y_black == 1)]
    Y_black_pos = Y_black[np.where(Y_black == 1)]
    Y_hat_black_pos = Y_hat_black[np.where(Y_black == 1)]  
    N_black_pos = X_black_pos.shape[0]
    Y_hat_black_pos_sum = np.sum(Y_hat_black_pos)
    X_black_pos_sum = np.sum(X_black_pos, axis=0)
    
    # Black, Negative
    X_black_neg = X_black[np.where(Y_black == 0)]
    Y_black_neg = Y_black[np.where(Y_black == 0)]
    Y_hat_black_neg = Y_hat_black[np.where(Y_black == 0)]  
    N_black_neg = X_black_neg.shape[0]
    Y_hat_black_neg_sum = np.sum(Y_hat_black_neg)
    X_black_neg_sum = np.sum(X_black_neg, axis=0)
    
    n = X.shape[1]
  
    FPR_gradient = (np.sum(np.multiply(np.reshape(np.repeat(Y_hat_white_neg*(1-Y_hat_white_neg), n, 0),(N_white_neg,n)),X_white_neg),axis=0) / N_white_neg) - (np.sum(np.multiply(np.reshape(np.repeat(Y_hat_black_neg*(1-Y_hat_black_neg), n, 0),(N_black_neg,n)),X_black_neg),axis=0) / N_black_neg)
    FNR_gradient = - (np.sum(np.multiply(np.reshape(np.repeat(Y_hat_white_pos*(1-Y_hat_white_pos), n, 0),(N_white_pos,n)),X_white_pos),axis=0) / N_white_pos) + (np.sum(np.multiply(np.reshape(np.repeat(Y_hat_black_pos*(1-Y_hat_black_pos), n, 0),(N_black_pos,n)),X_black_pos),axis=0) / N_black_pos)
   
  
    #print(FPR_gradient)
    #print(FNR_gradient)
    #for j in range(gradient.shape[0]):
    #    gradient[j] = ((Y_hat_white_pos_sum*X_black_pos_sum[j] - Y_hat_black_pos_sum*X_white_pos_sum[j]) / (Y_hat_black_pos_sum**2))
    
    #print(gradient)
    # Check whether we should use the gradient or anti-gradient
    #regularizer = (Y_hat_white_pos_sum / Y_hat_black_pos_sum) - (N_white_pos / N_black_pos)
    temp1 = perf_measure(Y_white, np.round(Y_hat_white))
    temp2 = perf_measure(Y_black, np.round(Y_hat_black))
    FPR_difference_rep = temp1[1]-temp2[1]
    FNR_difference_rep = temp1[3]-temp2[3]
    
    FPR_difference = (Y_hat_white_neg_sum / N_white_neg) - (Y_hat_black_neg_sum / N_black_neg)
    FNR_difference = ((N_white_pos-Y_hat_white_pos_sum) / N_white_pos) - ((N_black_pos - Y_hat_black_pos_sum) / N_black_pos)
    
    FPR_gradient = (-FPR_gradient, FPR_gradient)[FPR_difference < 0]
    FNR_gradient = (-FNR_gradient, FNR_gradient)[FNR_difference < 0]
    
    #print(FPR_difference, FPR_difference_rep)
    
    return [FPR_gradient,FNR_gradient, FPR_difference, FNR_difference, FPR_difference_rep, FNR_difference_rep]    
    

if __name__ == "__main__":
    
    data_list = create_two_year_files()
    X,y = set_data(data_list)
    
    print(X.shape)
     # White
    X_white = X[np.where(X[:,2] == 1)]
    Y_white = y[np.where(X[:,2] == 1)]
    #Y_hat_white = Y_hat[np.where(X[:,2] == 1)]
    
    # Black
    X_black = X[np.where(X[:,2] == 2)]
    Y_black = y[np.where(X[:,2] == 2)]
    #Y_hat_black = Y_hat[np.where(X[:,2] == 2)]
    
    # White, Positive
    X_white_pos = X_white[np.where(Y_white == 1)]
    Y_white_pos = Y_white[np.where(Y_white == 1)]
    #Y_hat_white_pos = Y_hat_white[np.where(Y_white == 1)]
    N_white_pos = X_white_pos.shape[0]
    #Y_hat_white_pos_sum = np.sum(Y_hat_white_pos)
    X_white_pos_sum = np.sum(X_white_pos, axis=0)
    
    # White, Negative
    X_white_neg = X_white[np.where(Y_white == 0)]
    Y_white_neg = Y_white[np.where(Y_white == 0)]
    #Y_hat_white_neg = Y_hat_white[np.where(Y_white == 0)]
    N_white_neg = X_white_neg.shape[0]
    #Y_hat_white_neg_sum = np.sum(Y_hat_white_neg)
    X_white_neg_sum = np.sum(X_white_neg, axis=0)
    
    # Black, Positive
    X_black_pos = X_black[np.where(Y_black == 1)]
    Y_black_pos = Y_black[np.where(Y_black == 1)]
    #Y_hat_black_pos = Y_hat_black[np.where(Y_black == 1)]  
    N_black_pos = X_black_pos.shape[0]
    #Y_hat_black_pos_sum = np.sum(Y_hat_black_pos)
    X_black_pos_sum = np.sum(X_black_pos, axis=0)
    
    # Black, Negative
    X_black_neg = X_black[np.where(Y_black == 0)]
    Y_black_neg = Y_black[np.where(Y_black == 0)]
    #Y_hat_black_neg = Y_hat_black[np.where(Y_black == 0)]  
    N_black_neg = X_black_neg.shape[0]
    #Y_hat_black_neg_sum = np.sum(Y_hat_black_neg)
    X_black_neg_sum = np.sum(X_black_neg, axis=0)
    
    print(N_white_pos)
    print(N_white_neg)
    print(N_black_pos)
    print(N_black_neg)
    
    
    #X_sen = np.zeros((10000,1))
    #y_synthetic = np.zeros(10000)
    #y_synthetic[0:5000] += 1
    
    #X_synthetic_1 = np.random.multivariate_normal([2, 2], [[3, 1], [1, 3]], 2500)
    #X_synthetic_2 = np.random.multivariate_normal([2, 2], [[3, 1], [1, 3]], 2500)
    #X_synthetic_3 = np.random.multivariate_normal([1, 1], [[3, 3], [1, 3]], 2500)
    #X_synthetic_4 = np.random.multivariate_normal([-2, -2], [[3, 1], [1, 3]], 2500)
    
    #X_sen[0:2500] = 1 + np.zeros((2500,1))
    #X_sen[2500:5000] = 2 + np.zeros((2500,1))
    #X_sen[5000:7500] = 1 + np.zeros((2500,1))
    #X_sen[7500:10000] = 2 + np.zeros((2500,1))
    
    
    #X_synthetic = np.concatenate((np.concatenate((X_synthetic_1,X_synthetic_2,X_synthetic_3,X_synthetic_4),axis=0),X_sen), axis=1)
    
    
    
    coeffs = np.zeros((5,5))
    
    errors_white = np.zeros((5,4))
    errors_black = np.zeros((5,4))
    acc_white = 0.0
    acc_black = 0.0
    
    errors_age1 = np.zeros((5,4))
    errors_age2 = np.zeros((5,4))
    errors_age3 = np.zeros((5,4))
    acc_age1 = 0.0
    acc_age2 = 0.0
    acc_age3 = 0.0
    total = np.zeros((5,6))
    acc = 0
    
    runs_train = []
    runs_test = []
    #coeffs_synthetic = zeros((5,5))
    #errors_white_synthetic = np.zeros((5,4))
    #errors_black_synthetic = np.zeros((5,4))
    #acc_white_synthetic = 0.0
    #acc_black_synthetic = 0.0
    
    for j in range(5):
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=j)
        #model2 = LogisticRegression()
        #model2.fit(X_train, y_train)
        
        coeffs[j,:],run_train,run_test = logistic_regression(X_train, y_train, X_test, y_test, num_steps = 200, learning_rate = 5e-5, add_intercept=False)
        
        runs_train.append(run_train)
        runs_test.append(run_test)
        
        predicted = np.round(sigmoid(np.dot(X_test,coeffs[j,:])))
        #predicted = model2.predict(X_test)
        #probs = model2.predict_proba(X_test)
        print(j)
        #acc += metrics.accuracy_score(y_test, predicted)
    #print metrics.roc_auc_score(y_test, probs[:, 1])
    #print(predicted)
    
        X_test_white = X_test[np.where(X_test[:,2] == 1)]
        y_test_white = y_test[np.where(X_test[:,2] == 1)]
        predicted_white =  predicted[np.where(X_test[:,2] == 1)]

    
        X_test_black = X_test[np.where(X_test[:,2] == 2)]
        y_test_black = y_test[np.where(X_test[:,2] == 2)]
        predicted_black =  predicted[np.where(X_test[:,2] == 2)]
    
        acc += (1 - (np.sum(np.abs(predicted-y_test))/y_test.shape[0]))
        acc_white += metrics.accuracy_score(y_test_white, predicted_white)
        acc_black += metrics.accuracy_score(y_test_black, predicted_black)
    
        #print(j)
        
        errors_white[j,:] = perf_measure(y_test_white, predicted_white)
        errors_black[j,:] = perf_measure(y_test_black, predicted_black)
    
        total[j,:] = [metrics.accuracy_score(y_test_white, predicted_white), metrics.accuracy_score(y_test_black, predicted_black), errors_white[j,1], errors_black[j,1], errors_white[j,3], errors_black[j,3]]
        
        # Age-based
        
        #X_test_age1 = X_test[np.where(X_test[:,0] == 1)]
        #y_test_age1 = y_test[np.where(X_test[:,0] == 1)]
        #predicted_age1 =  [predicted[i] for i in [np.where(X_test[:,0] == 1)]]

    
        #X_test_age2 = X_test[np.where(X_test[:,0] == 2)]
        #y_test_age2 = y_test[np.where(X_test[:,0] == 2)]
        #predicted_age2 =  [predicted[i] for i in [np.where(X_test[:,0] == 2)]]
    
        #X_test_age3 = X_test[np.where(X_test[:,0] == 3)]
        #y_test_age3 = y_test[np.where(X_test[:,0] == 3)]
        #predicted_age3 =  [predicted[i] for i in [np.where(X_test[:,0] == 3)]]
    
    
        #acc_age1 += metrics.accuracy_score(list(y_test_age1.transpose()[0]), list(predicted_age1[0]))
        #acc_age2 += metrics.accuracy_score(list(y_test_age2.transpose()[0]), list(predicted_age2[0]))
        #acc_age3 += metrics.accuracy_score(list(y_test_age3.transpose()[0]), list(predicted_age3[0]))
        
        #errors_age1[j,:] = perf_measure(list(y_test_age1.transpose()[0]), list(predicted_age1[0]))
        #errors_age2[j,:] = perf_measure(list(y_test_age2.transpose()[0]), list(predicted_age2[0]))
        #errors_age3[j,:] = perf_measure(list(y_test_age3.transpose()[0]), list(predicted_age3[0]))
        
    
    #print errors_white
    #print errors_black
    rates_white = (np.sum(errors_white, axis=0))/5
    rates_black = (np.sum(errors_black, axis=0))/5
    coeffs = np.log((np.sum(np.exp(coeffs), axis=0))/5)
    
    rates_age1 = (np.sum(errors_age1, axis=0))/5
    rates_age2 = (np.sum(errors_age2, axis=0))/5
    rates_age3 = (np.sum(errors_age3, axis=0))/5
    

    
    #print(np.sum(errors_white, axis=0))/100
    #print(np.sum(errors_black, axis=0))/100
    
    #print('')
    #print('acc on whites:')
    #print(acc_white/5)
    
    #print('acc on blacks:')
    #print(acc_black/5)
    
    #print('')
    #model = LogisticRegression()
    #model = model.fit(X, y)

    # check the accuracy on the training set
    #print(model.score(X, y))1
    #print('Coefficients:')
    #print(coeffs)
    
    print('')
    
    #print('FPR on whites:')
    #print(rates_white[1])
    #print('FPR on blacks:')
    #print(rates_black[1])
    #print('')
    #print('FNR on whites:')
    #print(rates_white[3])
    #print('FNR on blacks:')
    #print(rates_black[3])

    print('')
    print(total)
    print("Acc:")
    print(acc/5)
    #print("Avg. Acc:")
    #print((np.sum(total[:,0])/5)*(float(2103)/5278)+(np.sum(total[:,1])/5)*(float(3175)/5278))
    print("Avg. Acc White:")
    print(np.sum(total[:,0])/5)
    print("Avg. Acc Black:")
    print(np.sum(total[:,1])/5)
    print("Avg. FPR difference(Black-White):")
    print(np.sum(total[:,3]-total[:,2])/5)
    print("Avg. FNR difference(Black-white):")
    print(np.sum(total[:,5]-total[:,4])/5)
    
    
    run_train_avg = (runs_train[0]+runs_train[1]+runs_train[2]+runs_train[3]+runs_train[4])/5
    run_test_avg = (runs_test[0]+runs_test[1]+runs_test[2]+runs_test[3]+runs_test[4])/5
    
    plt.plot(run_train_avg[:,0],'r-',label="Accuracy (our method)",linewidth=3)
    plt.plot(run_train_avg[:,3],'r--',label="Accuracy (Standard logistic regression)",linewidth=3)
    plt.plot(run_train_avg[:,1],'g-',label="FPR difference (our method)",linewidth=3)
    plt.plot(run_train_avg[:,4],'g--',label="FPR difference (Standard LR)",linewidth=3)
    plt.plot(run_train_avg[:,2],'b-',label="FNR difference (our method)",linewidth=3)
    plt.plot(run_train_avg[:,5],'b--',label="FNR difference (Standard LR)",linewidth=3)
    plt.legend(loc='best',prop={'size':11},ncol=3)
    plt.xlabel('Number of iterations')
    plt.ylabel('Percentage')
    #plt.title('Per-iteration performance on training data')
    plt.show()
    
    plt.clf()
    #plotter(run_test_avg)
    plt.plot(run_test_avg[:,0],'r-',label="Accuracy (our method)",linewidth=3)
    plt.plot(run_test_avg[:,3],'r--',label="Accuracy (Standard logistic regression)",linewidth=3)
    plt.plot(run_test_avg[:,1],'g-',label="FPR difference (our method)",linewidth=3)
    plt.plot(run_test_avg[:,4],'g--',label="FPR difference (Standard LR)",linewidth=3)
    plt.plot(run_test_avg[:,2],'b-',label="FNR difference (our method)",linewidth=3)
    plt.plot(run_test_avg[:,5],'b--',label="FNR difference (Standard LR)",linewidth=3)
    plt.xlabel('Number of iterations')
    plt.ylabel('Percentage')
    leg = plt.legend(prop={'size':11},loc=2,ncol=1)
    if leg:
        leg.draggable()
    
    #plt.title('Per-iteration performance on test data') 
    plt.show()
    #print('FPR on age1:')
    #print(rates_age1[1])
    #print('FPR on age2:')
    #print(rates_age2[1])
    #print('FPR on age3:')
    #print(rates_age3[1])
    #print('')
    #print('FNR on age1:')
    #print(rates_age1[3])
    #print('FNR on age2:')
    #print(rates_age2[3])
    #print('FNR on age3:')
    #print(rates_age3[3])
    
    #print('')
    #print('acc on age1:')
    #print(acc_age1/100)
    
    #print('acc on age2:')
    #print(acc_age2/100)
    
    #print('acc on age3:')
    #print(acc_age3/100)
    
    #df = pd.DataFrame(np.c_[X,y], columns=['Age Category', 'Gender', 'Race', 'Number of Priors', 'Charge Degree', 'Is Recidivist'])
    #scatter_matrix(df, alpha=0.2, figsize=(6, 6))
    
    #X_white = X[np.where(X[:,2] == 1)]
    #y_white = y[np.where(X[:,2] == 1)]
    
    #X_black = X[np.where(X[:,2] == 1)]
    #y_black = y[np.where(X[:,2] == 1)]
    
    #X_white = np.delete(X_white, 2, 1)
    #X_black = np.delete(X_black, 2, 1)
    
    #plt.figure()
    #parallel_coordinates(X_white, ['Age Category', 'Gender', 'Number of Priors', 'Charge Degree'])
    
    #plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    print("DONE!")
    
    
    # Train\Test Separately
    
    X_white = X[np.where(X[:,2] == 1)]
    y_white = y[np.where(X[:,2] == 1)]
    
    X_black = X[np.where(X[:,2] == 2)]
    y_black = y[np.where(X[:,2] == 2)]
    
    coeffs_white = np.zeros((100,5))
    coeffs_black = np.zeros((100,5))
    
    errors_white = np.zeros((100,4))
    errors_black = np.zeros((100,4))
    acc_white = 0.0
    acc_black = 0.0
    
    for j in range(100):
        X_train_white, X_test_white, y_train_white, y_test_white = train_test_split(X_white, y_white, test_size=0.3, random_state=j)
        X_train_black, X_test_black, y_train_black, y_test_black = train_test_split(X_black, y_black, test_size=0.3, random_state=j)
        
        model3 = LogisticRegression()
        model3.fit(X_train_white, y_train_white)
        model4 = LogisticRegression()
        model4.fit(X_train_black, y_train_black)
        
    
        coeffs_white[j,:] = model3.coef_
        coeffs_black[j,:] = model4.coef_
        
        predicted_white = model3.predict(X_test_white)
        probs = model3.predict_proba(X_test_white)
        predicted_black = model4.predict(X_test_black)
        probs = model4.predict_proba(X_test_black)
        
        acc_white += metrics.accuracy_score(list(y_test_white), list(predicted_white))
        acc_black += metrics.accuracy_score(list(y_test_black), list(predicted_black))
    
        #print(j)
        
        errors_white[j,:] = perf_measure(list(y_test_white), list(predicted_white))
        errors_black[j,:] = perf_measure(list(y_test_black), list(predicted_black))


    rates_white = (np.sum(errors_white, axis=0))/100
    rates_black = (np.sum(errors_black, axis=0))/100
    coeffs_white = np.log((np.sum(np.exp(coeffs_white), axis=0))/100)
    coeffs_black = np.log((np.sum(np.exp(coeffs_black), axis=0))/100)

    print('')
    print('##################### WHITES ONLY ####################')
    print('')
    print('acc:')
    print(acc_white/100)
    
    print('Coefficients:')
    print(coeffs_white)
    
    print('')
    
    print('FPR:')
    print(rates_white[1])
    print('')
    print('FNR:')
    print(rates_white[3])


    print('')

    print('')
    print('##################### BLACKS ONLY ####################')
    
    print('')
    print('acc:')
    print(acc_black/100)
    
    print('')
    
    print('Coefficients:')
    print(coeffs_black)
    
    print('')
    
    print('FPR:')
    print(rates_black[1])
    print('')
    print('FNR:')
    print(rates_black[3])


    print('')
    
    
    coeffs_sep = np.zeros((2,5))
    coeffs_sep[0,:] = np.exp(coeffs_white)
    coeffs_sep[1,:] = np.exp(coeffs_black)
    
    print(coeffs_sep)
    
  



 
    #A bar plot with errorbars and height labels on individual bars
    
    N = 5

    ind = np.arange(N)  # the x locations for the groups
    width = 0.20       # the width of the bars

    
    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, coeffs_sep[0,:], width, color='y')
    rects2 = ax.bar(ind + width, coeffs_sep[1,:], width, color='k')
    rects3 = ax.bar(ind + 2*width, np.exp(coeffs), width, color='b')
    
    # add some text for labels, title and axes ticks
    ax.set_ylabel('exp-coefficients')
    ax.set_title('Logistic Regression exp-coefficients by Race')
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(('Age_Cat', 'Gender', 'Race', 'Num_Priors', 'Charge_Deg'))
    
    #ax.legend((rects1[0], rects2[0]), ('White', 'Black'))
    
    
    def autolabel(rects):
        """
        Attach a text label above each bar displaying its height
        """
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                    '%.2f' % float(height),
                    ha='center', va='bottom')
    
    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    
    plt.show()
        
    
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    


    
    
    
    
