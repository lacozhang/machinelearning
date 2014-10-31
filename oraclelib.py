#!/usr/bin/python

import re
import math
import codecs

def ngram_scores(ref, tr):

    match_cnt = 0
    for gram in ref:
        if gram in tr:
            match_cnt = match_cnt + min(ref[gram], tr[gram])
    return match_cnt



def ngram_match(ref, tr):
    
    scores = []
    for gram in zip(ref, tr):
        scores.append(ngram_scores(gram[0], gram[1]))
    return (sum(scores), scores)


def SentParse(ref_sent, trl_sent):
        #calculate ref ngrams
    ref_words = re.split(r'\s+', tokenization( ref_sent ))
    ref_ngrams = []
    ref_len = 0
    for j in range(1, 5):
        ngrams = word2ngrams(ref_words, j)
        ngram_dict = dict()
        for gram in ngrams:
            ngram_dict[gram] = 0

        for gram in ngrams:
            ngram_dict.setdefault(gram, 0)
            ngram_dict[gram] = ngram_dict[gram] + 1
        ref_ngrams.append(ngram_dict)
            

    for key in ref_ngrams[0]:
        ref_len = ref_len + ref_ngrams[0][key]
        #calculate trl ngrams
    trl_words = re.split(r'\s+', tokenization(trl_sent))
    trl_ngrams = []
    tst_cnt = []
    for j in range(1, 5):
        ngrams = word2ngrams(trl_words, j)
        tst_cnt.append(len(ngrams))
        ngram_dict = dict()
        for gram in ngrams:
            ngram_dict[gram] = 0
        for gram in ngrams:
            ngram_dict[gram] = ngram_dict[gram] + 1
        trl_ngrams.append( ngram_dict )
    (score, match_cnt) = ngram_match(ref_ngrams, trl_ngrams)
    return (score, match_cnt, tst_cnt, ref_len)


class Reader :
#trl, ref : [
#    "sent ", "", "sent"
#]
#used to compute match and test ngrams for a single translation engine
#which represented as a sequence of plain texts
    def __init__(self, trl, ref, tokenizer):

        self.trl = trl
        self.ref = ref
        self.tokenizer = tokenizer
        self.Parse()
        return None


    def bleu_score(self, ref, tr):
    
        scores = []
        for gram in zip(ref, tr):
            scores.append( self.ngram_scores(gram[0], gram[1]) )
        return (sum(scores), scores)


    def ngram_scores(self, ref, tr):

        match_cnt = 0
        for gram in ref:
            if gram in tr:
                match_cnt = match_cnt + min(ref[gram], tr[gram])
        return match_cnt

    def SentParse(self, ref_sent, trl_sent):
        #calculate ref ngrams
        ref_words = re.split(r'\s+', self.tokenizer( ref_sent ))
        ref_ngrams = []
	ref_len = 0
        for j in range(1, 5):
            ngrams = word2ngrams(ref_words, j)
            ngram_dict = dict()
            for gram in ngrams:
                ngram_dict[gram] = 0
            for gram in ngrams:
                ngram_dict.setdefault(gram, 0)
                ngram_dict[gram] = ngram_dict[gram] + 1
            ref_ngrams.append(ngram_dict)
            

        for key in ref_ngrams[0]:
            ref_len = ref_len + ref_ngrams[0][key]
        #calculate trl ngrams
        trl_words = re.split(r'\s+', self.tokenizer(trl_sent))
        trl_ngrams = []
        tst_cnt = []
        for j in range(1, 5):
            ngrams = word2ngrams(trl_words, j)
            tst_cnt.append(len(ngrams))
            ngram_dict = dict()
            for gram in ngrams:
                ngram_dict[gram] = 0
            for gram in ngrams:
                ngram_dict[gram] = ngram_dict[gram] + 1
            trl_ngrams.append( ngram_dict )
        (score, match_cnt) = self.bleu_score(ref_ngrams, trl_ngrams)
        return (score, match_cnt, tst_cnt, ref_len)


    def Parse(self):

        print "Sents : %d" % (len(self.ref))
        if len(self.trl) != len(self.ref):
            print "The number of trls : %d" % (len(self.trl))
            print "the number of ref : %d" % (len(self.ref))
            print "The number of translation doesn't match the number of reference"
            exit()

        self.all_match_cnt = []
        self.all_tst_cnt = []
        self.scores = []
        self.ref_len = []
        for i in range(len(self.trl)):
            ref_sent = self.ref[i]
            tr_sent = self.trl[i]
            (score, match_cnt, tst_cnt, ref_len) = self.SentParse(ref_sent,  tr_sent)
            self.all_match_cnt.append(match_cnt)
            self.all_tst_cnt.append(tst_cnt)
            self.scores.append(score)
            self.ref_len.append(ref_len)
        return None

    def Result(self):
        return (self.scores, self.all_match_cnt, self.all_tst_cnt, self.ref_len)


class MultipleRef:
    
    def __init__(self, tokenization, trl, *ref):
        self.trl = trl
        self.refs = ref
        self.tokenizer = tokenization
        self.Parse()
        return None

    def MultiSel(self, cnt, candidate):
        
        real_cnt = []
        for item in zip(cnt, candidate):
            tmp = [max(item[0][i], item[1][i]) for i in range(len(item[0]))]
            real_cnt.append(tmp)
        return real_cnt

    def ReflenSel(self, cnt, candidate):

        real_cnt = []
        for item in zip(cnt, candidate):
            tmp = max(item[0], item[1])
            real_cnt.append(tmp)
        return real_cnt

    def Parse(self):

        (real_score, real_match_cnt, real_tst_cnt, real_ref_len) = Reader(self.trl, self.refs[0], self.tokenizer).Result()

        for ref in self.refs[1:]:
            (tmp_score, tmp_match_cnt, tmp_tst_cnt, tmp_ref_len) = Reader(self.trl, ref, self.tokenizer).Result()
            real_match_cnt = self.MultiSel(real_match_cnt, tmp_match_cnt)
            real_tst_cnt = self.MultiSel(real_tst_cnt, tmp_tst_cnt)
            real_ref_len = self.ReflenSel(real_ref_len, tmp_ref_len)

        self.score = real_score
        self.match_cnt = real_match_cnt
        self.tst_cnt = real_tst_cnt
        self.ref_len = real_ref_len
        return None

    def Result(self):
        return (self.score, self.match_cnt, self.tst_cnt, self.ref_len)
            
def word2ngrams(text, cnt) :

    if 1 == cnt :
        return word2unigram(text)
    if 2 == cnt :
        return word2bigram(text)
    if 3 == cnt :
        return word2trigram(text)
    if 4 == cnt :
        return word24gram(text)
    return None

def word2unigram(text):
    return [(text[i],) for i in range(len(text))]

def word2bigram(text):
    return [(text[i], text[i+1]) for i in range(len(text)-1)]

def word2trigram(text):
    return [(text[i], text[i+1], text[i+2]) for i in range(len(text)-2)]

def word24gram(text):
    return [(text[i], text[i+1], text[i+2], text[i+3]) for i in range(len(text)-3)]
            

def tokenization( text ):

    text = re.subn(r'<skipped>',r'', text)[0]
    text = re.subn(r'-\n', r'', text)[0]
    text = re.subn(r'\n', r' ', text)[0]
    text = re.subn(r'&quot;',r'"', text)[0]
    text = re.subn(r'&amp;', r'&', text)[0]
    text = re.subn(r'&lt;', r'<', text)[0]
    text = re.subn(r'&gt;', r'>', text)[0]

    text = " "+ text + " "
    text = re.subn(r'([\{-\~\[-\` -\&\(-\+\:-\@\/])', r' \g<1> ', text)[0]
    text = re.subn(r'([^0-9])([\.,])', r'\g<1> \g<2> ', text)[0]
    text = re.subn(r'([\.,])([^0-9])', r' \g<1> \g<2>', text)[0]
    text = re.subn(r'([0-9])(-)', r'\g<1> \g<2> ', text)[0]
    text = re.subn(r'\s+', r' ', text)[0]
    text = re.subn(r'^\s+', r'', text)[0]
    text = re.subn(r'\s+$', r'', text)[0]
    return text

def extract_seg(text):
    pat = r'<seg\s+id="\d+">(.*?)<\/seg>'
    ret = re.findall(pat, text, re.I)
#    print len(ret)
    return tuple(ret)


def compute_bleu(ref_length, match_cnt, tst_cnt):
    score = float(0)
    iscore = float(0)
    exp_len_score = (float)
    if tst_cnt[0] > 0:
        exp_len_score = math.exp( min(0, 1 - float(ref_length)/float(tst_cnt[0])))
    smooth = float(1)
    realscore = float(0)

    for i in range(4):
        if tst_cnt[i] == 0:
            iscore = float(0)
        elif match_cnt[i] == 0:
            smooth = smooth * 2
            iscore = math.log(float(1) / float((smooth*tst_cnt[i])))
        else :
            iscore = math.log(float(match_cnt[i])/float(tst_cnt[i]))
        score += iscore
    realscore = math.exp(float(score)/float(4)) * exp_len_score
    return realscore

def accumulate(cnt):
    all = [0, 0, 0, 0]
    for item in cnt:
        all = [all[i] + item[i] for i in range(4)]
    return all

def read_text(name):
    text = ""
    for line in codecs.open(name, 'r', 'utf-8'):
        line = line.strip()
        text = text + line
    return text

def read_bleu_record(name):
    bleu = dict()
    for line in codecs.open(name, 'r', 'utf-8'):
        line = line.strip()
        tmp = re.split(r':', line, re.I)
        bleu[tmp[0]][tmp[1]] = float(tmp[2])
    return bleu


def join_plain(name):
    return ''.join(read_plain(name))

def read_plain(name):
    dat = []
    for line in codecs.open(name, 'r', 'utf-8'):
        line = line.strip()
        dat.append(line)
    return dat

def read_data(name):
    if re.search(r'sgm$', name, re.I):
        return extract_seg(join_plain(name))
    else:
        return read_plain(name)