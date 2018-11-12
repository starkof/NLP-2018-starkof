# A text classifier using logistic regression
# by: Stephan N. Ofosuhene

from sklearn.pipeline import Pipeline
from sklearn.utils import Bunch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from random import random
import numpy as np
import nltk


training_percent = 0.8

base_dir = '/Users/stephanofosuhene/Documents/Documents /Year 4 Sem 1/NLP/Labs/Lab_4/data/'
input_filenames = [base_dir + 'amazon_cells_labelled.txt', base_dir + 'imdb_labelled.txt', base_dir + 'yelp_labelled.txt']


def do_normalization(text):
    stemmer = nltk.LancasterStemmer()
    lemmatizer = nltk.WordNetLemmatizer()

    text = text.lower()

    text = ' '.join([stemmer.stem(s) for s in text.split(' ')])
    text = ' '.join([lemmatizer.lemmatize(s) for s in text.split(' ')])

    text = remove_punctuations(text)

    return text


def remove_punctuations(s):
    return s.replace(',', '').replace('.', '').replace('!', '').replace('?', '').replace('(', '').replace(
        ')', '').replace('"', '').replace('$', '').replace("'", '').replace(':', '').replace('*', '').replace('%', '')


def load_data(filenames, normalize=False):
    training_data = Bunch()
    training_data['data'] = []
    training_data['targets'] = []

    testing_data = Bunch()
    testing_data['data'] = []
    testing_data['targets'] = []

    for filename in filenames:
        with open(filename) as f:
            for line in f:
                line = line.lower()
                line = line.rstrip().split('\t')

                if normalize:
                    line[0] = do_normalization(line[0])

                if random() < training_percent:
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


def train(normalize=False):
    filenames = input_filenames
    logistic_model = Pipeline([('tfidf', TfidfVectorizer()),
                               ('clf', LogisticRegression())])

    training_data, test_data = load_data(filenames, normalize=normalize)

    logistic_model.fit(training_data.data, training_data.targets)

    predictions = logistic_model.predict(test_data.data)

    return logistic_model


def main(filenames):
    print('Calculating list of accuracies')
    accuracies = [train(normalize=False) for i in range(10)]

    print()
    print(sum(accuracies)/len(accuracies))


if __name__ == '__main__':
    main(input_filenames)
