# Friend-Affinity-Finder
import os
os.chdir('C:/Users/papa_ki_pari/Desktop/data sets')
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,roc_auc_score,confusion_matrix
df = pd.read_excel('Friendaffinity1.xlsx')
df.columns
df
df.apply(lambda x:sum(x.isnull()),axis=0)
df.describe()
features=df.iloc[:,:-1].values
from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values='NaN',strategy='median',axis=0)
imputer.fit(features[:,[0,1,2,3,4]])
features[:,[0,1,2,3,4]]=imputer.fit_transform(features[:,[0,1,2,3,4]])
df['Affinity'].fillna(df['Affinity'].mean(),inplace=True)
df.apply(lambda x:sum(x.isnull()),axis=0)
df.dtypes
corr=df.corr()
corr
df.hist('Agreeableness'),df.hist('Conscientiousness'),df.hist('Extraversion'),df.hist('EmotionalRange'),df.hist('Openness')
df['Agreeableness'] = df.Agreeableness.astype(int)
df['Conscientiousness'] = df.Conscientiousness.astype(int)
df['Extraversion']=df.Extraversion.astype(int)
df['EmotionalRange'] = df.EmotionalRange.astype(int)
df['Openness'] = df.Openness.astype(int)
df['Affinity'] = df.Affinity.astype(int)
df.dtypes
x = df[['Agreeableness','Conscientiousness','Extraversion','EmotionalRange','Openness',]].values
y = df['Affinity'].values
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)
lgsr=LogisticRegression(random_state=0)
lgsr.fit(x_train,y_train)
y_pred=lgsr.predict(x_test)
accuracy=accuracy_score(y_test,y_pred)
accuracy
conf_matrix=confusion_matrix(y_test,y_pred)
conf_matrix
tp=14
fn=6
fp=1
tn=10
sensitivity=tp/(tp+fn)
specivicity=tn/(tn+fp)
print(sensitivity)
print(specivicity)
from sklearn.tree import DecisionTreeClassifier
dtc_clf=DecisionTreeClassifier()
dtc_clf.fit(x_train,y_train)
dtc_y_test_pred=dtc_clf.predict(x_test)
print(accuracy_score(y_test,dtc_y_test_pred))
from sklearn.svm import SVC
sc =SVC(kernel='rbf')
sc_classifier=sc.fit(x_train,y_train)
svc_y_test=sc_classifier.predict(x_test)
svc_acc_test=accuracy_score(y_test,svc_y_test)
svc_acc_test
from sklearn.ensemble import RandomForestClassifier
rmf=RandomForestClassifier(n_jobs=2,n_estimators = 10,criterion = 'entropy',random_state=0)
rf_classi=RandomForestClassifier()
rf_classi.fit(x_train,y_train)
rf_classi_y_test_pred=rf_classi.predict(x_test)
rf_accu_test=accuracy_score(y_test,rf_classi_y_test_pred)
rf_accu_test
from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=5)
classifier.fit(x_train,y_train)
y_predict=classifier.predict(x_test)
accuracy=accuracy_score(y_test,y_predict)
print("the accuracy is:",accuracy)
from sklearn.naive_bayes import GaussianNB
clf=GaussianNB()
clf.fit(x_train,y_train)
GaussianNB()
y_pred=clf.predict(x_test)
accuracy_nb=accuracy_score(y_test,y_pred)
accuracy_nb
xnew=[[3,3,3,2,3]]
ynew=rf_classi.predict(xnew)
print(ynew)
xnew1=[[0,2,1,0,2]]
ynew1=rf_classi.predict(xnew1)
print(ynew1)
from sklearn.metrics import roc_auc_score
print(roc_auc_score(y_test,rf_classi_y_test_pred))
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,rf_classi_y_test_pred))
