# train models with available data and pickle the models
# by: Stephan N. Ofosuhene

from logistic_regression import logistic_regression_classifier_sklearn
from naive_bayes import naive_bayes_sklearn
import pickle


def koobi_ti():
    folder = 'models/'

    logistic_normalized = logistic_regression_classifier_sklearn.train(normalize=True, train_percent=1)
    pickle.dump(logistic_normalized, open(folder + 'logistic_n.pickle', 'wb'))

    logistic_raw_text = logistic_regression_classifier_sklearn.train(normalize=False, train_percent=1)
    pickle.dump(logistic_raw_text, open(folder + 'logistic_u.pickle', 'wb'))

    naive_bayes_normalized = naive_bayes_sklearn.train(normalize=True, train_percent=1)
    pickle.dump(naive_bayes_normalized, open(folder + 'naive_bayes_n.pickle', 'wb'))

    naive_bayes_raw_text = naive_bayes_sklearn.train(normalize=False, train_percent=1)
    pickle.dump(naive_bayes_raw_text, open(folder + 'naive_bayes_u.pickle', 'wb'))


koobi_ti()
