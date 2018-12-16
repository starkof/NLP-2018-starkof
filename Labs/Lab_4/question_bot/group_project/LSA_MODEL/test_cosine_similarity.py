from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


a = 'the first sentence'
b = 'the second sentence'


def cosine_sim(a, b):
    count_vec = CountVectorizer().fit_transform([a, b])
    tfidf_trans = TfidfTransformer(count_vec)
    return cosine_similarity(tfidf_trans.norm[0], tfidf_trans.norm[1])


print(cosine_sim(a, b))
