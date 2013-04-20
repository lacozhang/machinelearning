#!/usr/bin/python

import argparse

from sklearn.datasets import load_svmlight_file
import svmlight as SVC
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.naive_bayes import GaussianNB as GNB
from sklearn.naive_bayes import MultinomialNB as MNB
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report as cr
from sklearn.metrics import accuracy_score as accus
from sklearn import cross_validation
#from sklearn.linear_model import LogisticRegression as LR
import numpy as np
import codecs
import oraclelib
import re

parser = argparse.ArgumentParser()
parser.add_argument('-d', dest='train', required=True)
parser.add_argument('-t', dest='test', required=True)
parser.add_argument('-f', dest='fold', required=True)
parser.add_argument('-r', dest='report', required=True)
parser.add_argument('-c', dest='cnt', required=True, help='the number of candidates for each sentence')
ret = parser.parse_args()

def convert2svmlight(dat):

    output = []
    for line in dat:
        items = re.split(r'\s+', line)
        tmp = []

        score = float(items[0])
        qid = int(re.split(r':',items[1])[1])
        features = items[2:]
        formated_feature = []
        for f in features:
            index, val = re.split(r':', f)
            formated_feature.append(
                        (int(index), float(val))
                        )
        tmp.append(score)
        tmp.append(formated_feature)
        tmp.append(qid)
        output.append(tuple(tmp))
    return output 

def convert2pairwise(feature, target, syscnt):

    if feature.shape[0]%4 != 0:
        print "Error"
        exit()

    start = 0

    pw_feature = []
    pw_target = []
    while start < feature.shape[0]:
        end = start + syscnt
        section = feature[start:end]
        ngram = target[start:end]
        for i in range(0, syscnt):
            for j in range(0, syscnt):

                if i == j:
                    continue

                if ngram[i] == ngram[j]:
                    continue

                tmp = (section[i] - section[j]).todense().tolist()[0]
                pw_feature.append(tmp)
                if ngram[i] - ngram[j] > 0:
                    pw_target.append(1)
                else:
                    pw_target.append(-1)
        start = end
    ret =  [np.array(pw_feature), np.array(pw_target)]
    return ret

def data_split(target, mfolds):
    
    sp = cross_validation.StratifiedKFold(target, mfolds)
    return sp

def collect_data_qid(data_idx, train):

    output = []
    for item in train:
        if item[2] in data_idx:
            output.append(item)
    return output

def clf_test(model, test):

    ranking = []
    start = 0
    test_size = len(test)
    while start < test_size:

        end = start + 4

        features = test[start:end]
        scores = [0, 0, 0, 0, 0]
        for i in range(4):
            for j in range(i):
                clf_feature = features[i] - features[j]
                pred = model.predict(clf_feature)
                if pred > 0:
                    scores[i] += 1.0
                else:
                    scores[j] += 1.0

        ranking.extend(scores)
        start = end
    return ranking

def ranking_test(model, test):
    ranking = model.predict(test)
    return ranking

def output_ranking(dat, output):

    for item in dat:
        output.write('%0.6f\n'%(item))
    return None

def my_accus(feature, pred):

    output = {}
    for f, p in zip(feature, pred):
        qid = f[2]
        if qid not in output:
            output[qid] = {}
            output[qid]['real'] = []
            output[qid]['pred'] = []
        output[qid]['real'].append(f[0])
        output[qid]['pred'].append(p)

    input = output
    num_disagree = 0
    num_agree = 0
    num_total = 0
    for qid in input:
        real = input[qid]['real']
        pred = input[qid]['pred']
        for i in range(len(real)):
            for j in range(0, i):
                if ((real[i] < real[j]) and (pred[i] < pred[j])) or ((real[i]>real[j]) and (pred[i]>pred[j])):
                    num_agree += 1
                    num_total += 1
                elif real[i] == real[j]:
                    continue
                else:
                    num_disagree += 1
                    num_total += 1
    return float(num_agree)/float(num_total)


def my_cross_val_score(data_fold, train, c_p):

    scores = []
    for x, y in data_fold:
        data_x = collect_data_qid(x, train)
        data_y = collect_data_qid(y, train)
        model = SVC.learn(data_x, C=c_p, kernel='linear', type='ranking')
        pred = SVC.classify(model, data_y)
        scores.append(
            my_accus(data_y, pred)
            )
    return scores

def my_GridSearchCV(data_fold, train, param):

    best = 0
    best_c = None
    for c in param['C']:
        scores = my_cross_val_score(data_fold, train, c)
        if (sum(scores)/len(scores)) > best:
            best = (sum(scores)/len(scores))
            best_c = c
    return best_c

class mySVM(object):

    def __init__(self, model_):
        self.model = model_
        return None

    def predict(self, sample_):
        pred = SVC.classify(self.model, sample_)
        return pred

def SVM_experiment(data_fold, train, test, dumper):

    param = { 'C':[] }
    for i in range(-15, 15):
        param['C'].append(pow(2, i))
    c_best = my_GridSearchCV(data_fold, train, param)

    dumper.write("Classifier: SVM\n")
    dumper.write('Best Parameters: %f'%(c_best))

    model = SVC.learn(train, C=c_best, kernel='linear', type='ranking')
    ret = mySVM(model)
    pred = ranking_test(ret, test)
    output_ranking(pred, codecs.open('svm.ranking', 'w', 'utf-8'))
    return None

def NB_experiment(data_fold, train, test, dumper):

    print "Ready to find the Best Parameters for Naive Bayes"

    print 'Gaussian Naive Bayes'
    nb = GNB()
    print "fitting NaiveBayes Experiment"

    dumper.write('Classifier: Naive Bayes\n')
    scores = cross_validation.cross_val_score(nb, train[0], train[1], 
                                              cv = data_fold, score_func=accus)

    reports = "Accuracy on Train: %0.2f (+/- %0.2f)"%(scores.mean(), scores.std()/2)
    print reports

    dumper.write(reports+'\n')
    reports = " ".join(['%0.2f'%(item) for item in scores])
    dumper.write(reports+'\n')
    
    nb = GNB()
    nb.fit(train[0], train[1])
    
    pred = clf_test(nb, test)
    output_ranking(pred, codecs.open('nb.ranking', 'w', 'utf-8'))
    return None

def LR_experiment(data_fold, train, test, dumper):

    tuned_param = [
        { 'penalty':['l2'],
          'dual':[True],
          'C':[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
          'fit_intercept':['False'],
          'class_weight':['auto']},
        ]

    model = GridSearchCV(LR(), tuned_param, score_func = accus)
    model.fit(train[0], train[1], cv=data_fold)

    print "Best Parameters in training set"
    print model.best_estimator_
    print ""

    dumper.write('Classifier: Logistic Regression\n')
    reports = 'Best Parameter:%r'%(model.best_estimator_)
    dumper.write(reports + '\n')

    scores = cross_validation.cross_val_score( model, train[0], train[1],
                                               cv = data_fold, score_func = accus)
    reports = 'Accuracy on Train Set:%0.2f (+/- %0.2f)'%(scores.mean(), scores.std()/2)
    dumper.write(reports+'\n')
    reports = ' '.join(['%0.2f'%item for item in scores])
    dumper.write(reports+'\n')

    pred = clf_test(model, test)
    output_ranking(pred, codecs.open('lr.ranking', 'w', 'utf-8'))
    return None

def RFC_experiment(data_fold, train, test, dumper):


    tuned_param = [
        {'n_estimators':[]}
        ]

    for i in range(1, 20):
        tuned_param[0]['n_estimators'].append(i*10)

    print "Classifier: Random Forest Classifier"
    dumper.write("Classifier: Random Forest Classifier\n")
    model = GridSearchCV(RFC(n_estimators=10), tuned_param, score_func = accus)
    model.fit(train[0], train[1], cv = data_fold)
    
    print "best parameters found in training set"
    print model.best_estimator_
    print ""

    dumper.write('Best Parameters: %r\n'%(model.best_estimator_))

    scores = cross_validation.cross_val_score(model, train[0], train[1],
                                              cv = data_fold, score_func=accus)
    reports = "Accuracy on Train: %0.2f (+/- %0.2f)"%(scores.mean(), scores.std()/2)
    print reports
    dumper.write(reports + '\n')
    reports = " ".join(['%f'%(item) for item in scores])
    dumper.write(reports + '\n')

    pred = clf_test(model, test)
    output_ranking(pred, codecs.open('rfc.ranking', 'w', 'utf-8'))
    return None

if __name__ == "__main__":


    #data preprocessing    
    dev_f, dev_t = load_svmlight_file(ret.train)
    print 'converting training dataset'
    sys_cnt = int(ret.cnt)
    dev = convert2pairwise(dev_f, dev_t)

    test_f, test_t = load_svmlight_file(ret.test)
    test_f = test_f.todense()

    print 'spliting data into %s folds'%ret.fold
    dev_split = data_split(dev[1], int(ret.fold))

    #report file
    sink = codecs.open(ret.report, 'w', 'utf-8')

    NB_experiment(dev_split, dev, test_f, sink)
    RFC_experiment(dev_split, dev, test_f, sink)
#    LR_experiment(dev_split, dev, test_f, sink)

#    dev4svm = convert2svmlight(oraclelib.read_plain(ret.train))
#    test4svm = convert2svmlight(oraclelib.read_plain(ret.test))
#    devsplit4svm = cross_validation.ShuffleSplit(len(dev4svm), 
#                                                 n_iter=int(ret.fold), 
#                                                 test_size = float(1)/float(ret.fold), 
#                                                 random_state=404)
#    SVM_experiment(devsplit4svm, dev4svm, test4svm, sink)
    sink.close()

