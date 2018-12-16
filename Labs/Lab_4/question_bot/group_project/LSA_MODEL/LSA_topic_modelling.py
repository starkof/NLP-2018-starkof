# %%
import re
from gensim import models, corpora
from nltk import word_tokenize
from nltk.corpus import stopwords
from gensim import similarities

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from random import random
import pickle
import os
import sys

train_percent = 0.90

questions_file = 'Questions.txt'
answers_file = 'Answers.txt'
topics_file = 'Topics.txt'

NUM_TOPICS = 125
TOP_N = 26
STOPWORDS = stopwords.words('english')


def remove_punctuations(s):
    return s.replace(',', '').replace('.', '').replace('!', '').replace('?', '').replace('(', '').replace(
        ')', '').replace('"', '').replace('$', '').replace("'", '').replace(':', '').replace('*', '').replace('%', '')


# training data
def clean(s):
    if s == '\n':
        return ''

    s = s.rstrip()
    temp = s
    s = re.sub(r'^[0-9]+\.?\t? ?', '', s).rstrip().lower()
    if s == '':
        s = temp
    s = remove_punctuations(s)

    return s


def load_data(t_percent):
    training_data = {'questions': [], 'answers': [], 'topics': []}
    test_data = {'questions': [], 'answers': [], 'topics': []}
    with open(questions_file) as q, open(answers_file) as a, open(topics_file) as t:
        questions = [clean(l) for l in q if clean(l) != '']
        answers = [clean(l) for l in a if clean(l) != '']
        topics = [clean(l) for l in t if clean(l) != '']

    questions.pop()
    topics.pop()

    for i in range(len(questions)):
        # if i == 1000:
        #     break

        if random() < t_percent:
            training_data['questions'].append(questions[i])
            training_data['answers'].append(answers[i])
            training_data['topics'].append(topics[i])
        else:
            test_data['questions'].append(questions[i])
            test_data['answers'].append(answers[i])
            test_data['topics'].append(topics[i])

    return training_data, test_data


def argmax(topic_counts):
    max_topic = ''
    max_count = 0
    for topic in topic_counts:
        if topic_counts[topic] > max_count:
            max_count = topic_counts[topic]
            max_topic = topic

    return max_topic, max_count


def check_accuracy(test, training_data, dictionary, lsi_model, corpus):
    correct_count = 0
    for i in range(len(test['questions'])):
        topic = classify_topic(test['questions'][i], training_data, dictionary, lsi_model, corpus)

        print('predicted topic:', topic)
        print('actual topic:', test['topics'][i])

        if topic == test['topics'][i]:
            correct_count += 1

        print(correct_count/(i+1))
        print()

        # if n == 1:
        #     break

    return correct_count/len(test['questions'])


def clean_text(text):
    tokenized_text = word_tokenize(text.lower())
    cleaned_text = [t for t in tokenized_text if t not in STOPWORDS and re.match('[a-zA-Z\-][a-zA-Z\-]{2,}', t)]
    return cleaned_text


def get_topic(id, topic_bounds):
    for topic in topic_bounds:
        if topic_bounds[topic][0] <= id <= topic_bounds[topic][1]:
            return topic


def train(data):

    # For gensim we need to tokenize the data and filter out stopwords
    tokenized_data = []
    for text in data:
        tokenized_data.append(clean_text(text))

    # Build a Dictionary - association word to numeric id
    dictionary = corpora.Dictionary(tokenized_data)

    # Transform the collection of texts to a numerical form
    corpus = [dictionary.doc2bow(text) for text in tokenized_data]

    # Build the LSI model
    lsi_model = models.LsiModel(corpus=corpus, num_topics=NUM_TOPICS, id2word=dictionary)

    return lsi_model, dictionary, corpus


def classify_topic(text, training_data, dictionary, lsi_model, corpus):
    bow = dictionary.doc2bow(clean_text(text))

    # lsi_dist = lsi_model[bow]

    # print("=" * 20)

    lsi_index = similarities.MatrixSimilarity(lsi_model[corpus])

    # Let's perform some queries
    mat_similarities = lsi_index[lsi_model[bow]]

    # Sort the similarities
    mat_similarities = sorted(enumerate(mat_similarities), key=lambda item: -item[1])

    # print all similar documents
    count = {}
    for document_id, similarity in mat_similarities[:TOP_N]:
        doc_topic = training_data['topics'][document_id]

        if doc_topic not in count:
            count[doc_topic] = 1
        else:
            count[doc_topic] += 1

        max_topic, max_count = argmax(count)

    # print()
    # print('most likely topic:', max_topic)

    # print('topic counts:\n', count)

    return max_topic


def topic_accuracy(topn, test_data, training_data, dictionary, lsi_model, corpus):
    for i in range(20):
        accuracy = check_accuracy(test_data, training_data, dictionary, lsi_model, corpus)
        print('accuracy =', accuracy)

        with open('topic_accuracies.txt', 'a+') as f:
            print(NUM_TOPICS, topn, accuracy, file=f)

        topn += 5


def predict_answer(q, dictionary, lsi_model, corpus):
    bow = dictionary.doc2bow(clean_text(q))

    # lsi_dist = lsi_model[bow]

    # print("=" * 20)

    lsi_index = similarities.MatrixSimilarity(lsi_model[corpus])

    # Let's perform some queries
    mat_similarities = lsi_index[lsi_model[bow]]

    # Sort the similarities
    mat_similarities = sorted(enumerate(mat_similarities), key=lambda item: -item[1])

    answer_id = mat_similarities[0][0]

    # print('predicted answer id:', answer_id)

    return answer_id


def cosine_sim(a, b):
    count_vec = CountVectorizer().fit_transform([a, b])
    tfidf_trans = TfidfTransformer(count_vec)
    return cosine_similarity(tfidf_trans.norm[0], tfidf_trans.norm[1])[0][0]


def question_accuracy(training_data, test_data, dictionary, lsi_model, corpus):
    sims = 0
    correct = 0
    for i in range(len(test_data['questions'])):
        ans_id = predict_answer(test_data['questions'][i], dictionary, lsi_model, corpus)

        print('training questions:', len(training_data['questions']))
        print('training answers:', len(training_data['answers']))

        correct_ans = test_data['answers'][i]
        predicted_ans = training_data['answers'][ans_id]

        cosim = cosine_sim(correct_ans, predicted_ans)
        if cosim > 0.5:
            correct += 1

        # if predicted_ans == correct_ans:
        sims += cosim

        print('similarity:', cosim)
        print('average sim:', sims/2)
        print('accuracy:', correct/(i+1))
        print()

    with open('question_accuracy.txt', 'a+') as f:
        print('accuracy:', correct/len(test_data['questions']), file=f)

    print('accuracy:', correct/len(test_data['questions']))


def dev_test():
    training_data, test_data = load_data(train_percent)

    print('training data:', len(training_data['questions']))
    print('test data:', len(test_data['questions']))

    lsi_model, dictionary, corpus = train(training_data['questions'])

    model = (lsi_model, dictionary, corpus)

    with open('pickled_model.p', 'wb') as f:
        pickle.dump(model, f)

    # question_accuracy(training_data, test_data, dictionary, lsi_model, corpus)

    # pid = predict_answer('who is the head of state of ghana', dictionary, lsi_model, corpus)
    # print(training_data['answers'][pid])
    #
    topic_accuracy(TOP_N, test_data, training_data, dictionary, lsi_model, corpus)


def main(argv):
    training_data, test_data = load_data(1)

    if os.path.isfile('pickled_model.p'):
        print('Load trained model')
        with open('pickled_model.p', 'rb') as f:
            m = pickle.load(f)
    else:
        print('Train model once')
        m = train(training_data['questions'])
        with open('pickled_model.p', 'wb') as f:
            pickle.dump(m, f)

    lsi_model, dictionary, corpus = m

    filename = argv[2]
    option = argv[1]

    n = 0
    if option == 'topic':
        with open(filename) as f, open('topic_results.txt', 'w') as out:
            for line in f:
                n += 1
                line = clean(line)
                topic = classify_topic(line, training_data, dictionary, lsi_model, corpus)
                print(topic, file=out)
                print('case:', n)

    elif option == 'qa':
        with open(filename) as f, open('qa_results.txt', 'w') as out:
            for line in f:
                n += 1
                line = clean(line)
                answer = predict_answer(line, dictionary, lsi_model, corpus)
                print(training_data['answers'][answer], file=out)
                print('case:', n)


# dev_test()


if __name__ == '__main__':
    main(sys.argv)

