
# coding: utf-8

# In[ ]:


#Importing necessary libraries.
import sys
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression


# In[ ]:


#This function allows one to open a given pickle file and use a model that has been stored in it.
def load(file):
    with open(file,'rb') as f:
        trainedData = pickle.load(f)
    return trainedData


# In[ ]:


#This function takes a given amount of files and splits them into training and testing data.
#It uses pandas librabry to read, and join the data together as one.
#It also uses sklearn's count vectorizer to turn the data into an array of counts.
#The vectorizer is saved in a pickle file to enable easy reference instead of constant preprocessiong after every run.
#The training and test data are then returned.
def preProcessing(*files):
    dataList = []
    for file in files:
        df = pd.read_csv(file, sep="\t", header = None)
        dataList.append(df)
    frame = pd.concat(dataList)
    X = frame[0]
    y = frame[1]
    vectorizer = CountVectorizer(analyzer = 'word',lowercase = False)
    features = vectorizer.fit_transform(X)
    with open("vectorizer1.pickle",'wb') as f:
        pickle.dump(vectorizer, f)
    featureArr = features.toarray()
    X_train, X_test, y_train, y_test = train_test_split(featureArr, y, test_size = 0.2, random_state = 0)
    return X_train, X_test, y_train, y_test

#This function takes a given amount of files to be classified.
#It uses pandas librabry to read, and join the data together as one.
#It also uses sklearn's count vectorizer to turn the data into an array of counts.
#The vectorizer is saved in a pickle file to enable easy reference instead of constant preprocessiong after every run.
#The training and test data are then returned. 
def toVector(*files):
    dataList = []
    
    for file in files:
        df = pd.read_table(file, header = None)
        dataList.append(df)
        
    frame = pd.concat(dataList)
    
    vectorizer = load('vectorizer1.pickle')
    features = vectorizer.transform(frame[0])
    featureArr = features.toarray()
    return featureArr


# In[ ]:


#This function creates a logistic regression model and trains on x_train and y_train.
#The trained model is stored in a pickle file.
def logRegression(tup):
    regressor = LogisticRegression()
    regressor.fit(tup[0], tup[2])
    with open("regressor1.pickle",'wb') as f:
        pickle.dump(regressor, f)


# In[ ]:


#This function supplies data to be trained on to the preProcessing fuction 
#It also runs the logistic regression function above on the results of the preProcessing function.
def train():
    values = preProcessing('amazon_cells_labelled.txt','yelp_labelled.txt','imdb_labelled.txt')
    logRegression(values)


# In[ ]:


#This function allows command line interface by taking the argument variables parameter.
#It vectorizes the data to be classified (argv[3]).
#It then checks if the model has been saved in a pickle file, if it is, it loads its contents and calls the predict method on the 
#given file. If it hasn't been saved, the model is trained and the pickle file generated from training is used in predicting.
#It then writes the preidcted content to a .txt file
def main(argv):
    testData = toVector(argv[3])
    writeFile = open('lr-u.txt', 'w')
    exist = False
    try:
        f = open('regressor1.pickle', 'rb')
        f.close()
        exist = True
    except FileNotFoundError:
        pass
    if exist is True:
        model = load('regressor1.pickle')
        pred = model.predict(testData)
        pred = pred.tolist()
        for i in pred:
            writeFile.write(str(i)+'\n')
    else:
        train()
        model = load('regressor1.pickle')
        pred = model.predict(testData)
        pred = pred.tolist()
        for i in pred:
            writeFile.write(str(i)+'\n')
main(sys.argv) 


# In[ ]:


if __name__ == 'main':
    main(argv)

