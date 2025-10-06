import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score,accuracy_score,roc_auc_score,classification_report,confusion_matrix,roc_curve

df=pd.read_csv("d:\\cssv\\spam.csv", encoding="latin1")
pd.options.display.max_columns=None
df.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1,inplace=True)

df.columns=['spam','SMS']
x=df.drop("spam",axis=1)

vec=CountVectorizer()
X=vec.fit_transform(x['SMS'])
y=df['spam']
Y=y.values
enc=LabelEncoder()
Y=enc.fit_transform(Y)

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=0.8,random_state=42)

model=MultinomialNB()
model.fit(X_train,Y_train)
y_pred=model.predict(X_test)
pro=model.predict_proba(X_test)
print("Accuracy score ",accuracy_score(Y_test,y_pred))
print("Confussion Matrix",confusion_matrix(Y_test,y_pred))
print('Classification report',classification_report(Y_test,y_pred))
print('f1 score',f1_score(Y_test,y_pred))
print("AUC",roc_auc_score(Y_test,y_pred))
#---------------------plotting the RoC curve--------------
fpr,tpr,threshold=roc_curve(Y_test,pro[:,1])
plt.plot(fpr,tpr,label='ROC Curve',color='red',ls=':')
plt.plot([0,1],[0,1],label='Middle',color='black',ls='--')
plt.legend(loc='best')
plt.grid()
plt.show()