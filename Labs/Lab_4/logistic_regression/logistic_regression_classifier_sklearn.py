# A text classifier using logistic regression
# by: Stephan N. Ofosuhene

from sklearn.pipeline import Pipeline
from sklearn.utils import Bunch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score

from random import random
import numpy as np
import nltk


base_dir = '/Users/stephanofosuhene/Documents/Documents /Year 4 Sem 1/NLP/Labs/Lab_4/data/'
input_filenames = [base_dir + 'amazon_cells_labelled.txt', base_dir + 'imdb_labelled.txt', base_dir + 'yelp_labelled.txt']


def do_normalization(text):
    stemmer = nltk.LancasterStemmer()
    lemmatizer = nltk.WordNetLemmatizer()

    # text = text.lower()

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


def train(normalize=False, train_percent=0.8, return_accuracy=False):
    print('Training Logistic Regression model. Normalize =', normalize)
    filenames = input_filenames
    logistic_model = Pipeline([('tfidf', TfidfVectorizer()),
                               ('clf', LogisticRegression())])

    training_data, test_data = load_data(filenames, normalize=normalize, train_percent=train_percent)

    logistic_model.fit(training_data.data, training_data.targets)

    if return_accuracy:
        predictions = logistic_model.predict(test_data.data)
        accuracy = np.mean(predictions == test_data.targets)

        avg_precision = average_precision_score(test_data.targets, predictions)
        # print('Average precision:', avg_precision)

        recall = recall_score(test_data.targets, predictions)
        # print('Recall:', recall * 100)

        return accuracy, recall

    return logistic_model


def test_model():
    metrics = [train(normalize=True, return_accuracy=True) for i in range(10)]

    accuracy = 0
    recall = 0
    for a, p in metrics:
        accuracy += a
        recall += p

    print()
    print('Average accuracy:', accuracy/len(metrics) * 100)
    print('Average recall:', recall/len(metrics) * 100)
    print()

    metrics = [train(normalize=False, return_accuracy=True) for i in range(10)]

    accuracy = 0
    recall = 0
    for a, p in metrics:
        accuracy += a
        recall += p

    print()
    print('Average accuracy:', accuracy / len(metrics) * 100)
    print('Average recall:', recall/len(metrics) * 100)


test_model()
