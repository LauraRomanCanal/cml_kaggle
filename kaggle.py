import sklearn
import xgboost as xgb
import numpy as np
import os
import pandas as pd
import matplotlib as plt
from sklearn import cross_validation as cv
from sklearn import metrics

os.chdir('/home/euan/documents/comp-machine-learn/kaggle')

X_test = pd.read_csv('X_test.dat')
X_train = pd.read_csv('X_train.dat')
Y_train = pd.read_csv('y_train.dat',header=None)
sample = pd.read_csv('sample.dat')

X_test.columns = range(0,79)
X_train.columns = range(0,79)
Y_train.columns = range(0,2)

# TESTING IN SAMPLE
x_train,x_test,y_train,y_test = cv.train_test_split(X_train,Y_train,test_size=0.2,random_state=1234)
x_train = pd.DataFrame(x_train.as_matrix()[:,1:78])
x_test = pd.DataFrame(x_test.as_matrix()[:,1:78])

model = xgb.XGBClassifier()
model.fit(x_train,y_train[1])

y_pred = model.predict(x_test)
predictions = [round(value) for value in y_pred]
auc = metrics.roc_auc_score(y_true = y_test[1],y_score = predictions)
print('AUC: %.2f%%' % (auc*100))

# FITTING THE MODEL

#remove the ID column from the training and test data
X_train_data = pd.DataFrame(X_train.as_matrix()[:,1:78])
X_test_data = pd.DataFrame(X_test.as_matrix()[:,1:78])

model = xgb.XGBClassifier()
model.fit(X_train_data,Y_train[1])

y_pred = model.predict(X_test_data)
predictions = pd.DataFrame( [round(value) for value in y_pred])

preds = pd.concat((X_test[0],predictions), axis = 1)
preds.columns = ['Id','Prediction']

pd.DataFrame.to_csv(preds,path_or_buf='predictions.dat',index=False)
