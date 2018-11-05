# Naive bayes model implemented with NLTK
# by: Stephan N. Ofosuhene

import nltk
from nltk.tokenize import word_tokenize
from random import random

training_percent = 0.8


#%%
def remove_punctuations(s):
    return s.replace(',', '').replace('.', '').replace('!', '').replace('?', '').replace('-', '').replace('(',
                                                                                                          '').replace(
        ')', '').replace('"', '').replace('$', '').replace("'", '').replace(':', '').replace('*', '').replace('%', '')


def load_data(filename):
    training_data = []
    testing_data = []

    with open(filename) as f:
        for line in f:
            line = line.lower()
            line = remove_punctuations(line)
            line = tuple(line.rstrip().split('\t'))
            if random() > training_percent:
                training_data.append(line)
            else:
                testing_data.append(line)
    all_words = set(word.lower() for passage in training_data for word in word_tokenize(passage[0]))
    t = [({word: (word in word_tokenize(x[0])) for word in all_words}, x[1]) for x in training_data]
    print(t)

    return training_data, testing_data


def main():
    training_set, testing_set = load_data('data/amazon_cells_labelled.txt')
    print(training_set)

    classifier = nltk.NaiveBayesClassifier.train(training_set)

    print('Accuracy:', nltk.classify.accuracy(classifier, testing_set) * 100)


if __name__ == '__main__':
    main()
