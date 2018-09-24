#!/usr/bin/env python
# coding: utf-8

# In[1]:


from math import log
from random import random
import string
import sys
import os

# setup required before program runs
positive_cls = 1
negative_cls = 0

# proportion of the training data set (as a decimal) aside for testing
testing_proportion = 0.2  # set to 0 if only training is required


def remove_punctuations(s):
    return s.replace(',', '').replace('.', '').replace('!', '').replace('?', '').replace('-', '').replace('(', '').replace(')', '').replace('"', '').replace('$','').replace("'", '').replace(':', '').replace('*', '').replace('%', '')


def randomize_datasource(filenames):
    # separates and writes the cases into different files to make the
    # program more ram efficient
    training = open('training.txt', 'w')
    testing = open('testing.txt', 'w')
    
    for fn in filenames:
        with open(fn) as f:
            for line in f:
                line = line.lower()
                line = remove_punctuations(line)
                if random() > testing_proportion:
                    print(line, file=training, end='')
                else:
                    print(line, file=testing, end='')
    testing.close()
    training.close()


def add_to_vocabulary(doc, vocab):
    words = doc.split(' ')
    
    for word in words:
        if word != '':
            if word not in vocab:
                vocab[word] = 1
            else:
                vocab[word] += 1


# In[4]:


def find_log_likelyhood(word, cls_vocab, cls_den):
    if word in cls_vocab:
        n = cls_vocab[word]
    else:
        n = 0
        
    return log((n + 1)/cls_den)


def calc_denominator(vocab, cls):
    s = 0
    for word in vocab['all']:
        if word in vocab[cls]:
            s += vocab[cls][word] + 1

    return s


def train_naive_bayes_classifier(files):
    n_positive_docs = 0
    n_negative_docs = 0
    log_likelyhood = {positive_cls: {}, negative_cls: {}}
    for filename in files:
        with open(filename) as f:
            # a dictionary of dictionaries to hold all vocabulary
            vocabulary = {'all': {}, positive_cls: {}, negative_cls: {}}

            for line in f:
                record = line.rstrip().split('\t')
                text = record[0]
                cls = int(record[1])

                add_to_vocabulary(text, vocabulary['all'])

                if cls == negative_cls:
                    n_negative_docs += 1
                    add_to_vocabulary(text, vocabulary[negative_cls])
                elif cls == positive_cls:
                    n_positive_docs += 1
                    add_to_vocabulary(text, vocabulary[positive_cls])
                
        n_docs = n_positive_docs + n_negative_docs
        
        logprior = {}
        logprior[negative_cls] = log(n_negative_docs/n_docs)
        logprior[positive_cls] = log(n_positive_docs/n_docs)
                
        positive_den = calc_denominator(vocabulary, positive_cls)
        negative_den = calc_denominator(vocabulary, negative_cls)
        
        for word in vocabulary['all']:
            log_likelyhood[positive_cls][word] = find_log_likelyhood(word, vocabulary[positive_cls], positive_den)
            log_likelyhood[negative_cls][word] = find_log_likelyhood(word, vocabulary[negative_cls], negative_den)
            
    print('Number of positive training docs:', n_positive_docs)
    print('Number of negative training docs:', n_negative_docs)
    print('Total number of training docs:', n_docs)
    
    return log_likelyhood, logprior, vocabulary


def argmax(sum_dict):
    if sum_dict[positive_cls] > sum_dict[negative_cls]:
        return positive_cls
    else:
        return negative_cls


# the doc should be a list of words
def classify_doc(log_likelyhood, logprior, vocabulary, filename):
    with open(filename) as f:
        n = 0
        k = 0
        outfile = open('output.txt', 'w')
        for line in f:
            record = line.rstrip().split('\t')
            line = record[0]
            c = int(record[1])
            doc = line.split(' ')
            classes = [positive_cls, negative_cls]
            s = {}

            s[positive_cls] = logprior[positive_cls]
            s[negative_cls] = logprior[negative_cls]
            for cls in classes:
                for word in doc:
                    if word in vocabulary['all']:
                        s[cls] += log_likelyhood[cls][word]
            print(argmax(s), file=outfile)
            if (argmax(s) == c):
                n += 1
            k += 1
        outfile.close()
        
        print('Number of test docs', k)
        if k == 0:
            return None
        print('Accuracy (%)', (n/k)*100)


def main(argv):
    randomize_datasource(argv[1:])
    log_likelyhood, logprior, vocabulary = train_naive_bayes_classifier(['training.txt'])
    classify_doc(log_likelyhood, logprior, vocabulary, 'testing.txt')

    os.remove('training.txt')
    os.remove('testing.txt')


if __name__ == '__main__':
    main(sys.argv)

