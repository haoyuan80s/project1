# TODO: OOP modulize code

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import preprocessing 
import utils
reload(utils)
from utils import *


print "loading data...."
train, test, macro  = (pd.read_csv('data/train.csv'),
                       pd.read_csv('data/test.csv'),
                       pd.read_csv('data/macro.csv'))
id_test = test.id  # for writting output file

def feature_engineering():
    print "feature engereering...." # drop time, since test is all "future"
    y_train, x_train, x_test = (train["price_doc"],
                                train.drop(["id", "timestamp", "price_doc"], axis=1),
                                test.drop(["id", "timestamp"], axis=1))
    for c in x_train.columns:
        if x_train[c].dtype == 'object':
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(x_train[c].values)) 
            x_train[c] = lbl.transform(list(x_train[c].values))        
    for c in x_test.columns:
        if x_test[c].dtype == 'object':
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(x_test[c].values)) 
            x_test[c] = lbl.transform(list(x_test[c].values))
    return  np.array(x_train), np.array(y_train) # X,y for training

X, y = feature_engineering()
dfun = DataFun(X, y)

params = {
    'eta': 0.02,
    'max_depth': 5,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1}
dfun.train_gradient_boost(params, num_boost_round = 4) # try early stopping
print "training score = ", rmlse(dfun.predict(X),y)

# dtest = xgb.DMatrix(x_test)
# dtrain = xgb.DMatrix(x_train.head, y_train.head)
# model = xgb.train(xgb_params, dtrain, num_boost_round= 500) # train with all data
# print "Outputting...."
# y_predict = model.predict(dtest)
# output = pd.DataFrame({'id': id_test, 'price_doc': y_predict})
# output.head()
# output.to_csv('xgbSub.csv', index=False)
# print 'done!'


# draft


# early stopping
# http://xgboost.readthedocs.io/en/latest/python/python_intro.html
# cv_output = xgb.cv(params, dtrain, num_boost_round=1000, early_stopping_rounds=20,
#    verbose_eval=0, show_stdv=False)
# cv_output[['train-rmse-mean', 'test-rmse-mean']].plot()
# num_boost_rounds = len(cv_output)
