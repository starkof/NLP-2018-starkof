# Implementation of naive bayes classifier with sklearn
# by: Stephan N. Ofosuhene

from random import random
from sklearn.utils import Bunch
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import numpy as np


training_percent = 0.2
input_file = '/Users/stephanofosuhene/Documents/Documents /Year 4 Sem 1/NLP/Labs/Lab_4/data/amazon_cells_labelled.txt'


def remove_punctuations(s):
    return s.replace(',', '').replace('.', '').replace('!', '').replace('?', '').replace('-', '').replace('(',
                                                                                                          '').replace(
        ')', '').replace('"', '').replace('$', '').replace("'", '').replace(':', '').replace('*', '').replace('%', '')


def load_data(filename):
    training_data = Bunch()
    training_data['data'] = []
    training_data['targets'] = []

    testing_data = Bunch()
    testing_data['data'] = []
    testing_data['targets'] = []

    with open(filename) as f:
        for line in f:
            line = line.lower()
            line = remove_punctuations(line)
            line = tuple(line.rstrip().split('\t'))

            if random() > training_percent:
                training_data.data.append(line[0])
                training_data.targets.append(int(line[1]))
            else:
                testing_data.data.append(line[0])
                testing_data.targets.append(int(line[1]))

    training_data.data = np.array(training_data.data)
    training_data.targets = np.array(training_data.targets)

    testing_data.data = np.array(testing_data.data)
    testing_data.targets = np.array(testing_data.targets)

    return training_data, testing_data


def main(filename):
    training_data, testing_data = load_data(filename)

    count_vect = CountVectorizer()
    train_counts = count_vect.fit_transform(training_data.data)

    tfid_transformer = TfidfTransformer(use_idf=False).fit(train_counts)
    train_tfid = tfid_transformer.transform(train_counts)

    trained_model = MultinomialNB().fit(train_tfid, training_data.targets)

    test_counts = count_vect.transform(testing_data.data)
    test_tfid = tfid_transformer.transform(test_counts)

    predicted = trained_model.predict(test_tfid)

    n = 0
    for label, prediction in zip(testing_data.targets, predicted):
        if label == prediction:
            n += 1

    print('Training accuracy:', np.mean(predicted == testing_data.targets)*100)


main(input_file)
