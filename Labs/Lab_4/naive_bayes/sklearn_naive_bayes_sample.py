from sklearn import datasets
from sklearn.naive_bayes import GaussianNB

iris = datasets.load_iris()
gnb = GaussianNB()


# print('Iris data:', len(iris.data), len(iris.target))
print(iris)
print()

model = gnb.fit(iris.data, iris.target)
predictions = model.predict(iris.data)

print('Predictions:', predictions)

print("Number of mislabeled points out of a total {0} points : {1}".format(iris.data.shape[0], (iris.target != predictions).sum()))
