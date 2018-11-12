# Lab 4 for Natural Language Processing at Ashesi University
# by: Stephan N. Ofosuhene

import numpy as np
import pickle
import sys


def read_test_data(filename):
    data = []

    with open(filename) as f:
        for line in f:
            line = line.rstrip()

            data.append(line)

    return np.array(data)


def write_to_file(predictions, filename):
    with open(filename, 'w') as f:
        for i in predictions:
            print(i, file=f)


def main(argv):
    classifier = argv[1]
    version = argv[2]
    test_filename = argv[3]

    test_data = np.array([])

    pickle_folder = 'models/'

    try:
        test_data = read_test_data(test_filename)
    except FileNotFoundError:
        print('File not found')
        exit(-1)

    out_filename = 'results-'+classifier + '-' + version + '-version.txt'

    if classifier == 'nb':

        if version == 'u':
            model = pickle.load(open(pickle_folder + 'naive_bayes_u.pickle', 'rb'))
            predictions = model.predict(test_data)
            write_to_file(predictions, out_filename)

        elif version == 'n':
            model = pickle.load(open(pickle_folder + 'naive_bayes_n.pickle', 'rb'))
            predictions = model.predict(test_data)
            write_to_file(predictions, out_filename)

        else:
            print('Invalid input combination')

    elif classifier == 'lr':
        if version == 'u':
            model = pickle.load(open(pickle_folder + 'logistic_u.pickle', 'rb'))
            predictions = model.predict(test_data)
            write_to_file(predictions, out_filename)

        elif version == 'n':
            model = pickle.load(open(pickle_folder + 'logistic_n.pickle', 'rb'))
            predictions = model.predict(test_data)
            write_to_file(predictions, out_filename)

        else:
            print('Invalid input combination')

    else:
        print('Invalid input')


if __name__ == '__main__':
    main(sys.argv)
