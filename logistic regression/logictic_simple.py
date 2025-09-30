import numpy as np 
import matplotlib.pylab as plt
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,roc_auc_score,roc_curve,confusion_matrix
from sklearn.linear_model import LogisticRegression


df=pd.read_csv('d:\\cssv\\logistic.csv')
X=df[['Age','Salary','Experience']]
Y=df['Purchased']
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=0.2,random_state=42)
sca=StandardScaler()
X_test_sca=sca.fit_transform(X_test)
X_train_sca=sca.transform(X_train)

model=LogisticRegression()
mod=model.fit(X_train_sca,Y_train)
y_pred=model.predict(X_test_sca)
pro=model.predict_proba(X_test_sca)
acc=accuracy_score(Y_test,y_pred)
con=confusion_matrix(Y_test,y_pred)
auc=roc_auc_score(Y_test,pro[:, 1])
#------predic my value -------------
new=np.array([[55,60000,15]])
new_sca=sca.transform(new)
new_pred=model.predict(new_sca)

#-----------------plotting the roc curve------------------
fpr, tpr, thresholds = roc_curve(Y_test, pro[:,1])
plt.plot(fpr, tpr, label="ROC Curve")
plt.plot([0,1], [0,1], 'k--', label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC Curve")
plt.legend()
plt.show()
