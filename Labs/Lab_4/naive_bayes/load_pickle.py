import pickle
import numpy as np


model = pickle.load(open('pickled_bayes.p', 'rb'))

docs = ['this is good', 'this is bad', "I'm not sure"]

print(model.predict(np.array(docs)))
