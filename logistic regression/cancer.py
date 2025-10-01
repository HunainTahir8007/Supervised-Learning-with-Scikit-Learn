import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score,confusion_matrix,roc_auc_score,roc_curve
from sklearn.linear_model import LogisticRegression

pd.options.display.max_columns=None 
dat=load_breast_cancer()
df = pd.DataFrame(dat.data, columns=dat.feature_names)
df['target']=dat.target

X=df
Y=df['target']
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=0.8,random_state=42)
#now we scale the values 
scale=StandardScaler()
X_train_sca=scale.fit_transform(X_train)
X_test_sca=scale.transform(X_test)
model=LogisticRegression(max_iter=5000)
model=model.fit(X_train_sca,Y_train)
y_pred=model.predict(X_test_sca)
proba=model.predict_proba(X_test_sca)
acc=accuracy_score(Y_test,y_pred)
con=confusion_matrix(Y_test,y_pred)
auc=roc_auc_score(Y_test,proba[:,1])
#---------------------plotting the roc curve ---------------
fpr,tpr,threshold=roc_curve(Y_test,proba[:,1])
plt.plot(fpr,tpr,label='Roc curve',color='red',ls='--')
plt.plot([0,1],[0,1],color='black',label='Random Guess')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate ")
plt.title("ROC Curve")
plt.legend()
plt.show()