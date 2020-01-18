# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 12:34:32 2020
@author: devil may cry
"""
import quandl as qd
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as ny
import math
from datetime import datetime
from sklearn import preprocessing, svm, model_selection
from sklearn.linear_model import LinearRegression
import pickle
df=qd.get('WIKI/GOOGL')

df=df[['Adj. High','Adj. Low','Adj. Open','Adj. Close','Adj. Volume']]
df['PCT']=(df['Adj. Close']-df['Adj. Open'])/df['Adj. Open']*100
df['HL_PCT']=(df['Adj. High']-df['Adj. Close'])/df['Adj. Close']*100

df=df[['HL_PCT', 'PCT','Adj. Volume','Adj. Close']]

pred_col='Adj. Close'
df.fillna(-99999,inplace=True)
pred_out=int(math.ceil(0.01*len(df)))
print(pred_out)

df['label']=df[pred_col].shift(-pred_out)

#Xa=ny.array(df.drop(['label'],1))
#ya=ny.array(df['label'])

#df.dropna(inplace=True)

X=ny.array(df.drop(['label'],1))
X=preprocessing.scale(X)

X=X[:-pred_out]
X_lately=X[-pred_out+1:]
df.dropna(inplace=True)

#y=df.iloc[:,df['label']
y = ny.array(df['label'])

X_train, X_test, y_train,y_test=model_selection.train_test_split(X,y,test_size=0.2)


# =============================================================================
# clf=LinearRegression(n_jobs=-1)
# clf.fit(X_train,y_train)
# with open('linear_regression.pickle','wb') as tmp:
#     pickle.dump(clf, tmp)
#     
# =============================================================================
pickle_in = open('linear_regression.pickle','rb')

clf=pickle.load(pickle_in)

accuracy=clf.score(X_test,y_test)
print(accuracy)
# predicting unknown values

pred_testset=clf.predict(X_test)

pred_set=clf.predict(X_lately)
#print(pred_set.head())

#display the results in graphical form 
# =============================================================================
# 
# plt.scatter(X_train[:,1], y_train)
# plt.plot(X_train, clf.predict(X_train), color='blue')
# plt.title('Stock price per day')
# plt.xlabel('Date and time')
# plt.ylabel('Stock Price')
# plt.show()
# =============================================================================
#predction graph

df['pred']=ny.nan

last_date=df.iloc[-1].name
first_unix=last_date.timestamp()
day=88600
next_unix = first_unix + day

for i in pred_set:
    
    next_date=datetime.fromtimestamp(next_unix)
    next_unix=next_unix+day
    df.loc[next_date]= [ny.nan for _ in range(len(df.columns)-1)] +[i] 
    
from matplotlib import style
style.use('ggplot')

df['Adj. Close'].plot()
df['pred'].plot()
plt.legend(loc=4)
plt.xlabel('date')
plt.ylabel('price')
plt.show()

