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


training_percent = 0.2
base_dir = '/Users/stephanofosuhene/Documents/Documents /Year 4 Sem 1/NLP/Labs/Lab_4/data/'
input_files = [base_dir + 'amazon_cells_labelled.txt', base_dir + 'imdb_labelled.txt', base_dir + 'yelp_labelled.txt']

# TODO: implement commandline parameters requirement
# TODO: use a library for text regularisation


def remove_punctuations(s):
    return s.replace(',', '').replace('.', '').replace('!', '').replace('?', '').replace('(', '').replace(
        ')', '').replace('"', '').replace('$', '').replace("'", '').replace(':', '').replace('*', '').replace('%', '')


def load_data(filenames):
    training_data = Bunch()
    training_data['data'] = []
    training_data['targets'] = []

    testing_data = Bunch()
    testing_data['data'] = []
    testing_data['targets'] = []

    stemmer = nltk.LancasterStemmer()
    lemmatizer = nltk.WordNetLemmatizer()

    for filename in filenames:
        print(filename)
        with open(filename) as f:
            for line in f:
                line = line.lower()
                line = remove_punctuations(line)
                line = line.rstrip().split('\t')
                # print('before lemmatization and stemming ->', line[0])

                # perform stemming
                line[0] = ' '.join([stemmer.stem(s) for s in line[0].split(' ')])

                # lemmatization
                line[0] = ' '.join([lemmatizer.lemmatize(s) for s in line[0].split(' ')])

                # print('after lemmatization and stemming ->', line[0])
                # print()

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


def raw_implementation(filename):
    training_data, test_data = load_data(filename)

    count_vect = CountVectorizer()
    train_counts = count_vect.fit_transform(training_data.data)

    tfid_transformer = TfidfTransformer(use_idf=False).fit(train_counts)
    train_tfid = tfid_transformer.transform(train_counts)

    trained_model = MultinomialNB().fit(train_tfid, training_data.targets)

    test_counts = count_vect.transform(test_data.data)
    test_tfid = tfid_transformer.transform(test_counts)

    predicted = trained_model.predict(test_tfid)

    n = 0
    for label, prediction in zip(test_data.targets, predicted):
        if label == prediction:
            n += 1

    print('Training accuracy:', np.mean(predicted == test_data.targets)*100)


def pipeline_implementation(input_filenames):
    # implements the naive bayes classifier using a pipeline to simplify the code
    text_classifier = Pipeline([('vect', CountVectorizer()),
                                ('tfidf', TfidfTransformer()),
                                ('clf', MultinomialNB())])

    training_data, test_data = load_data(input_filenames)

    text_classifier.fit(training_data.data, training_data.targets)

    predictions = text_classifier.predict(test_data.data)

    print('Model accuracy', np.mean(predictions == test_data.targets)*100)


# raw_implementation(input_files)
pipeline_implementation(input_files)
