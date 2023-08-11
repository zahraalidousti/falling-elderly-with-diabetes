import numpy  as np 
import pandas as pd 
from  sklearn.neural_network  import  MLPClassifier
from  sklearn.model_selection import  cross_val_score
from  sklearn.model_selection import  train_test_split
from  sklearn.metrics  import  confusion_matrix
from  sklearn.metrics  import  classification_report
import warnings
import tensorflow as tf

#algorithm
clf = MLPClassifier() 

#clf.fit(X_train, y_train,epochs=10)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_pred
y_pred == y_test
print(classification_report(y_test, y_pred))  
confusion_matrix(y_test, y_pred)
clf.score(X_train, y_train)
clf.score(X_test, y_test)
scores = cross_val_score(clf, X, y, cv=3, scoring='accuracy') # baraye arzyabi model
scores
scores.mean()
