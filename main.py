# TODO: 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import ensemble, preprocessing#, cross_validation
import xgboost as xgb

#from sklearn.utils import shuffle
#from sklearn.metrics import mean_absolute_error
#from sklearn.metrics import mean_squared_error

def RMSL(P,A):
    return np.sqrt(sum((np.log(P + 1) - np.log(A+1))**2)/len(P))

print "loading data...."
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
macro = pd.read_csv('data/macro.csv')
id_test = test.id

print "feature engereering...."
y_train = train["price_doc"]
x_train = train.drop(["id", "timestamp", "price_doc"], axis=1)
x_test = test.drop(["id", "timestamp"], axis=1)


for c in x_train.columns:
    if x_train[c].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(x_train[c].values)) 
        x_train[c] = lbl.transform(list(x_train[c].values))
        #x_train.drop(c,axis=1,inplace=True)
        
for c in x_test.columns:
    if x_test[c].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(x_test[c].values)) 
        x_test[c] = lbl.transform(list(x_test[c].values))
        #x_test.drop(c,axis=1,inplace=True)  


#import pdb; pdb.set_trace()
print "model training...."
xgb_params = {
    'eta': 0.05,
    'max_depth': 5,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}


dtrain = xgb.DMatrix(x_train, y_train)
dtest = xgb.DMatrix(x_test)

model = xgb.train(xgb_params, dtrain, num_boost_round= 500)
y_predict = model.predict(dtest)

output = pd.DataFrame({'id': id_test, 'price_doc': y_predict})
output.head()

output.to_csv('xgbSub.csv', index=False)

print 'done't
