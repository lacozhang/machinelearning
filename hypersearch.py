#!/usr/bin/python

import argparse
import svmlight as SVC
import codecs
import oraclelib
from sklearn import cross_validation
import re

parser = argparse.ArgumentParser()
parser.add_argument('-d', dest='train', required=True)
parser.add_argument('-f', dest='fold', required=True)
parser.add_argument('-r', dest='report', required=True)
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

def collect_data_qid(data_idx, train):

    output = []
    for item in train:
        if item[2] in data_idx:
            output.append(item)
    return output

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

def SVM_experiment(data_fold, train, dumper):

    param = { 'C':[] }
    for i in range(-5, 11):
        param['C'].append(pow(2, i))
    c_best = my_GridSearchCV(data_fold, train, param)

    print "Best Parameters: %0.8f"%c_best
    dumper.write("Classifier: SVM\n")
    dumper.write('Best Parameters: %f'%(c_best))
    return None

if __name__ == "__main__":


    #report file
    sink = codecs.open(ret.report, 'w', 'utf-8')

    dev4svm = convert2svmlight(oraclelib.read_plain(ret.train))
    devsplit4svm = cross_validation.ShuffleSplit(len(dev4svm), 
                                                 n_iter=int(ret.fold), 
                                                 test_size = float(1)/float(ret.fold), 
                                                 random_state=404)
    SVM_experiment(devsplit4svm, dev4svm, sink)

    sink.close()
