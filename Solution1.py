# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import read_excel

# Importing the dataset
dataset = read_excel('DA_Test_Modified.xlsx')

# Cleaning the texts
import re
import nltk
corpus = []

#Customized texts
for i in range(0, 7695):
    review = re.sub('[^a-zA-Z.&:]', ' ', dataset['Particulars'][i])
    corpus.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 100)
X = cv.fit_transform(corpus).toarray()
y = dataset['First Level Classification'].values

# Encoding the categories into numbers
from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)   

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred_classification_names=pd.DataFrame(labelencoder_y.inverse_transform(y_pred))
y_actualTest_classification_names=pd.DataFrame(labelencoder_y.inverse_transform(y_test))

#Accuracy
Accuracy=str(round(len(y_pred[y_pred==y_test])/len(y_pred)*100,2)) +'%'

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print('\n\n'+'Model Accuracy is ' + Accuracy+'\n\n')
print('To check results, please compare variable "y_actualTest_classification_names" with "y_pred_classification_names"'+'\n\n')

