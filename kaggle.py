import sklearn
import xgboost as xgb
import numpy as np
import os
import pandas as pd
import matplotlib as plt
from sklearn import cross_validation as cv
from sklearn import metrics
from sklearn import ensemble

os.chdir('/home/euan/documents/comp-machine-learn/cml_kaggle')

# Load Data
X_test = pd.read_csv('X_test.dat')
X_train = pd.read_csv('X_train.dat')
Y_train = pd.read_csv('y_train.dat',header=None)
sample = pd.read_csv('sample.dat')

# Rename column names
X_test.columns = range(0,79)
X_train.columns = range(0,79)
Y_train.columns = range(0,2)

# Remove the ID column from the training and test data
X_train_data = pd.DataFrame(X_train.as_matrix()[:,1:78])
X_test_data = pd.DataFrame(X_test.as_matrix()[:,1:78])

#seed=12345
#X_train_data_rand = X_train_data.sample(frac=1,random_state=seed)
#Y_train_rand = Y_train.sample(frac=1,random_state=seed)

####################################
# Model Selection
####################################

###########
# XGBOOST #
###########

# Cross validated AUC metric

model = xgb.XGBClassifier()
roc = metrics.make_scorer(metrics.roc_auc_score)
auc_xgboost = cv.cross_val_score(model, X_train_data,Y_train[1], scoring=roc, cv=5)

print('AUC cross-validated xgboost: %.2f%%' % (np.mean(auc_xgboost)*100)) #AUC ~72%

######################
# AdaBoostClassifier #
######################

# Cross validated AUC metric

model = ensemble.AdaBoostClassifier()
auc_adaboost = cv.cross_val_score(model, X_train,Y_train[1], scoring=roc, cv=5)
print('AUC: %.2f%%' % (np.mean(auc_adaboost)*100)) #AUC 79.66%

##############
# RandomForest
##############

# Cross validated AUC metric

model = ensemble.RandomForestClassifier()
auc_random_forest = cv.cross_val_score(model, X_train,Y_train[1], scoring=roc, cv=5)
print('AUC RF: %.2f%%' % (np.mean(auc_random_forest)*100)) # AUC 76.33%

########################
# Support Vector Machine
########################

# Cross validated AUC metric

model = ensemble.RandomForestClassifier()
auc_random_forest = cv.cross_val_score(model, X_train,Y_train[1], scoring=roc, cv=5)
print('AUC RF: %.2f%%' % (np.mean(auc_random_forest)*100)) # AUC 76.33%


########################################################
# Fitting our chosen model and classifying the test data
########################################################

model = xgb.XGBClassifier()
model.fit(X_train_data,Y_train[1])

y_pred = model.predict(X_test_data)
predictions = pd.DataFrame( [round(value) for value in y_pred])

preds = pd.concat((X_test[0],predictions), axis = 1)
preds.columns = ['Id','Prediction']

pd.DataFrame.to_csv(preds,path_or_buf='predictions.csv',index=False)
