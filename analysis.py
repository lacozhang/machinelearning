#!/usr/bin/python

import argparse
import codecs
import oraclelib
import scipy.stats as stat
import re


parser = argparse.ArgumentParser()
parser.add_argument("-f", required=True, dest="feature", help="feature files")
parser.add_argument("-p", required=True, dest="predict", help="prediction produced by svm-light")
parser.add_argument("-o", required=True, dest='output', help='file name used to store the results')
parser.add_argument('-na', required=True, dest='na', help='file names used to store nbest accuracy')
parser.add_argument('-c',  required=True, dest='cnt', help='N-Best of N')
ret = parser.parse_args()

def convert2intern(feature, predict):
    
    input = {}
    for (f, p) in zip(feature, predict):

        fields = re.split(r'\s+', f)

        rscore = float(fields[0])
        pscore = float(p)

#get qid from feature file
        idx = fields[1]
        if not re.match(r'qid:\d+', idx):
            print 'Error, format error %s'%(f)
            exit()

        qid = int(re.search(r'\d+', idx).group())
#insert data into input
        if not input.has_key(qid):
            input[qid] = dict()
            input[qid]['real'] = list()
            input[qid]['pred'] = list()

        input[qid]['real'].append(rscore)
        input[qid]['pred'].append(pscore)
    return input

def classify_accuracy(input):

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
    print "computing kendall's tau"
    print float(num_agree-num_disagree)/float(num_total)
    print "Classification Accurarcy"
    print "number of agree %d"%(num_agree)
    print "number of total %d"%(num_total)
    print float(num_agree)/float(num_total)
    return None

def spearman_r(input, output):
    for qid in input:
        real = input[qid]['real']
        pred = input[qid]['pred']
        (rho, p) = stat.spearmanr(real, pred)
        output.write('%f\n'%(rho))
    return None

def nbest_accuracy(input, output, n):

    total = 0
    for qid in input:
        real = input[qid]['real']
        pred = input[qid]['pred']
        sorted_real = sorted(real)
        sorted_pred = sorted(pred)

        sorted_real.reverse()
        sorted_pred.reverse()

        error = 0
        for val in sorted_real[:n]:
            cpos = real.index(val)
            cval = pred[cpos]
            if sorted_pred.index(cval) >= n:
                error += 1 

        output.write('%f\n'%(float(n-error)/float(n)))
        if error == 0:
            total += 1

    print '%d best accuracy number %d'%(n, total)
    return None


if __name__ == '__main__':
    
    feature = oraclelib.read_plain(ret.feature)
    predict = oraclelib.read_plain(ret.predict)
    input = convert2intern(feature, predict)
    output = codecs.open(ret.output, 'w', 'utf-8')

    classify_accuracy(input)
    spearman_r(input, output)
    output.close()

    output = codecs.open(ret.na, 'w', 'utf-8')
    nbest_accuracy(input, output, int(ret.cnt))
    output.close()


