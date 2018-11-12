# Implementation of naive bayes classifier with sklearn
# by: Stephan N. Ofosuhene

from random import random
from sklearn.utils import Bunch
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import numpy as np
import nltk

base_dir = '/Users/stephanofosuhene/Documents/Documents /Year 4 Sem 1/NLP/Labs/Lab_4/data/'
input_files = [base_dir + 'amazon_cells_labelled.txt', base_dir + 'imdb_labelled.txt', base_dir + 'yelp_labelled.txt']


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


def load_data(filenames, normalize=False, train_percent=0.8):
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

                if random() < train_percent:
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


def train(normalize=False, train_percent=0.8):
    print('Training Naive Bayes model. Normalize =', normalize)
    input_filenames = input_files

    # implements the naive bayes classifier using a pipeline to simplify the code
    text_classifier = Pipeline([('vect', CountVectorizer()),
                                ('tfidf', TfidfTransformer()),
                                ('clf', MultinomialNB())])

    training_data, test_data = load_data(input_filenames, normalize=normalize, train_percent=train_percent)

    text_classifier.fit(training_data.data, training_data.targets)

    return text_classifier

