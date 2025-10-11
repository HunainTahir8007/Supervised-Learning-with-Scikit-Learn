import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,roc_auc_score,roc_curve
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
pd.options.display.max_columns=None
df=pd.read_csv("d:\\cssv\\heart.csv")
X=df.drop('target',axis=1)
Y=df['target']

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=0.8,random_state=42)

sca=StandardScaler()
X_test_sca=sca.fit_transform(X_test)
X_train_sca=sca.transform(X_train)

model=SVC(kernel='linear',probability=True,class_weight='balanced')
mod=model.fit(X_train_sca,Y_train)
y_pred=model.predict(X_test_sca)
pro=model.predict_proba(X_test_sca)
print('Accuracy :: ',accuracy_score(Y_test,y_pred))
print('Classification Report : ',classification_report(Y_test,y_pred))
print("Confusion Matrix :",confusion_matrix(Y_test,y_pred))
print('Auc Score : ',roc_auc_score(Y_test,pro[:,1]))

#-------------------------testing----------------------------
my_test=np.array([[58,1,0,114,318,0,2,140,0,4.4,0,3,1]])
new_sca=sca.transform(my_test)
testing=model.predict(new_sca)
print(testing)
#---------------------------ploting --------------------------
fpr,tpr,threshold=roc_curve(Y_test,pro[:,1])
plt.plot(fpr,tpr,label='Roc Curve ',color='red',ls=":")
plt.plot([0,1],[0,1],label='Sepration line ',color='black')
plt.xlabel("False rate ")
plt.ylabel("True rate")
plt.grid(color='grey',linewidth=0.3,ls=':')
plt.legend()
plt.show()