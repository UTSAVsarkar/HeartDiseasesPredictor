import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

dataset = pd.read_csv('heart.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#Naive Bayes Classifier
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

infProb = classifier.predict_proba([[58,1,2,140,211,1,0,165,0,0,2,0,2]])
print(infProb[0][1]*100)

#creating the file to save the model
file = open('model.pkl', 'wb')
pickle.dump(classifier, file)
file.close()