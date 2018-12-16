# %%
import re
from gensim import models, corpora
from nltk import word_tokenize
from nltk.corpus import stopwords
from gensim import similarities

import nltk
from random import random

train_percent = 0.8

questions_file = 'Questions.txt'
answers_file = 'Answers.txt'
topics_file = 'Topics.txt'

NUM_TOPICS = 125
STOPWORDS = stopwords.words('english')


def do_normalization(text):
    stemmer = nltk.LancasterStemmer()
    lemmatizer = nltk.WordNetLemmatizer()

    # removing this line improve the performance of the classifier after normalization
    text = text.lower()

    text = ' '.join([stemmer.stem(s) for s in text.split(' ')])
    text = ' '.join([lemmatizer.lemmatize(s) for s in text.split(' ')])

    text = remove_punctuations(text)

    return text


def remove_punctuations(s):
    return s.replace(',', '').replace('.', '').replace('!', '').replace('?', '').replace('(', '').replace(
        ')', '').replace('"', '').replace('$', '').replace("'", '').replace(':', '').replace('*', '').replace('%', '')


# training data
def clean(s):
    if s == '\n':
        return ''

    temp = s
    s = re.sub(r'^[0-9]+\.?\t? ?', '', s).rstrip().lower()
    if s == '':
        s = temp.rstrip()
    s = remove_punctuations(s)

    return s


def load_data():
    training_data = {'questions': [], 'answers': [], 'topics': []}
    test_data = {'questions': [], 'answers': [], 'topics': []}
    with open(questions_file) as q, open(answers_file) as a, open(topics_file) as t:
        questions = [clean(l) for l in q if clean(l) != '']
        answers = [clean(l) for l in a if clean(l) != '']
        topics = [clean(l) for l in t if clean(l) != '']

    questions.pop()
    topics.pop()

    for i in range(len(questions)):
        if random() < train_percent:
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


def check_accuracy():
    pass


training_data, test_data = load_data()

print('training data:', len(training_data['questions']))
print('test data:', len(test_data['questions']))

# print('topic boundaries\n', topic_boundaries)

data = training_data['questions']

# NO_DOCUMENTS = len(data)
# print(NO_DOCUMENTS)
# print(data[:5])


def clean_text(text):
    tokenized_text = word_tokenize(text.lower())
    cleaned_text = [t for t in tokenized_text if t not in STOPWORDS and re.match('[a-zA-Z\-][a-zA-Z\-]{2,}', t)]
    return cleaned_text


def get_topic(id, topic_bounds):
    for topic in topic_bounds:
        if topic_bounds[topic][0] <= id <= topic_bounds[topic][1]:
            return topic


# For gensim we need to tokenize the data and filter out stopwords
tokenized_data = []
for text in data:
    tokenized_data.append(clean_text(text))

# Build a Dictionary - association word to numeric id
dictionary = corpora.Dictionary(tokenized_data)


# Transform the collection of texts to a numerical form
corpus = [dictionary.doc2bow(text) for text in tokenized_data]

# Have a look at how the 20th document looks like: [(word_id, count), ...]
# print(corpus[20])

# Build the LSI model
lsi_model = models.LsiModel(corpus=corpus, num_topics=NUM_TOPICS, id2word=dictionary)


text = 'what is blockchain'
bow = dictionary.doc2bow(clean_text(text))

lsi_dist = lsi_model[bow]

# print(lsi_dist)

s = 0
for x, y in lsi_dist:
    s += y
print(s)

print("=" * 20)

lsi_index = similarities.MatrixSimilarity(lsi_model[corpus])

# %%
# Let's perform some queries
similarities = lsi_index[lsi_model[bow]]

# Sort the similarities
similarities = sorted(enumerate(similarities), key=lambda item: -item[1])

# Top most similar documents:
# similarities.sort(key=lambda t: t[0], reverse=False)
print('similarities\n', similarities[:10])
print()
print('question len:', len(training_data['questions']))
print('answer len:', len(training_data['answers']))
print('topic len:', len(training_data['topics']))
print()

print()
print('doc check')
print(training_data['answers'][1260])

# print all similar documents
count = {}
n = 0
for document_id, similarity in similarities[:1]:
    print()
    print('document id =', document_id)
    print('question:', training_data['questions'][document_id])
    print('answer:', training_data['answers'][document_id])
    doc_topic = training_data['topics'][document_id]
    print('topic:', doc_topic)

    if doc_topic not in count:
        count[doc_topic] = 1
    else:
        count[doc_topic] += 1

    max_topic, max_count = argmax(count)

print()
print('most likely topic:', max_topic)

print('topic counts:\n', count)
