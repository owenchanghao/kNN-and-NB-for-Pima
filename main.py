import math
import numpy as np
from scipy.stats import norm

def classify_nn(training_filename, testing_filename, k):
    ## OPEN FILE
    training_file = open(training_filename)
    training_set = []
    for line in training_file:
        training_set.append(line.strip().split(','))
    training_file.close()
    testing_file = open(testing_filename)
    testing_set = []
    for line in testing_file:
        testing_set.append(line.strip().split(','))
    testing_file.close()
    ## PREDICT RESULT
    result = []
    for i in testing_set:
        votebox = []
        for j in training_set:
            n = 0
            diff = 0
            while n < len(i):
                diff += (float(j[n]) - float(i[n])) ** 2
                n += 1
            temp_dist = diff ** 0.5
            votebox.append([temp_dist, j[-1]])
        votebox.sort()
        votes = votebox[0:k]
        num_y = 0
        num_n = 0
        for vote in votes:
            if vote[1] == 'yes':
                num_y += 1
            elif vote[1] == 'no':
                num_n += 1
        if num_y > num_n:
            result.append('yes')
        elif num_y < num_n:
            result.append('no')
        else:
            result.append(votes[0])
    print(result)
    return result


def pdf(x, sigma, miu):
    return (1 / (sigma * math.sqrt(2 * math.pi))) * (math.e ** (- ((x - miu) ** 2) / (2 * (sigma ** 2))))

def classify_nb(training_filename, testing_filename):
    ## OPEN FILE
    training_file = open(training_filename)
    training_set = []
    for line in training_file:
        training_set.append(line.strip().split(','))
    training_file.close()
    testing_file = open(testing_filename)
    testing_set = []
    for line in testing_file:
        testing_set.append(line.strip().split(','))
    testing_file.close()
    ## CALCULATE NUMERICS
    y_averages = []
    n_averages = []
    y_stdivs = []
    n_stdivs = []
    n = 0
    while n < (len(training_set[0])-1):
        l_y = []
        l_n = []
        for item in training_set:
            if item[-1] == 'yes':
                l_y.append(float(item[n]))
            elif item[-1] == 'no':
                l_n.append(float(item[n]))
            else:
                print("Warning!")
        y_averages.append(np.mean(l_y))
        n_averages.append(np.mean(l_n))
        y_stdivs.append(np.var(l_y))
        n_stdivs.append(np.var(l_n))
        n += 1
    ## PREDICT RESULT
    result = []
    for i in testing_set:
        count = 0
        y_score = 1
        n_score = 1
        while count < len(i):
            y_score *= norm.pdf([float(i[count]), float(y_stdivs[count]), float(y_averages[count])])
            n_score *= norm.pdf([float(i[count]), float(n_stdivs[count]), float(n_averages[count])])
            count += 1
        if y_score >= n_score:
            result.append('yes')
        else:
            result.append('no')
    #print(result)
    return result


from sklearn.model_selection import StratifiedKFold
def stra_cv(file):
    ## OPEN FILE
    training_file = open(file)
    training_set = []
    for line in training_file:
        training_set.append(line.strip().split(','))
    training_file.close()
    partdata = []
    ## EXTRACT LABEL
    labels = []
    for i in training_set:
        labels.append(i[-1])
        #i.pop(-1)
        partdata.append(i[0:-1])
    ## STRATIFIED CROSS-VALIDATION
    skf = StratifiedKFold(n_splits = 10)
    X = np.array(partdata)
    y = np.array([i for i in labels])
    ## WRITE-IN DATA
    f = open('answer-stratified10folds.csv', 'w')
    count = 1
    for train_index, test_index in skf.split(X, y):
        #print("TRAIN:", train_index, "TEST:", test_index)
        print("fold", count, file=f)
        for i in test_index:
            print(','.join(str(i) for i in training_set[i]), file=f)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        count += 1
    f.close()

#from sklearn.model_selection import cross_val_score



classify_nn('pima.csv', 'train.csv', 5)
#classify_nb('pima.csv', 'train.csv')
#stra_cv('pima_test.csv')
