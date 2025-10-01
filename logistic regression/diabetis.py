import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from sklearn.metrics import accuracy_score,confusion_matrix,roc_auc_score,roc_curve,classification_report
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression
pd.options.display.max_columns=None
df=pd.read_csv("d:\\cssv\\diabetes.csv")
df.drop(columns=['SkinThickness','BloodPressure'],axis=1,inplace=True)
med=df['Insulin'].median()
df['Insulin'] = df['Insulin'].replace(0, med)
X=df.drop(['Outcome'],axis=1)
Y=df['Outcome']

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=0.8,random_state=42,)

scalar=StandardScaler()
X_train_sca=scalar.fit_transform(X_train)
X_test_sca=scalar.transform(X_test)

model=LogisticRegression(max_iter=5000,class_weight='balanced')
model=model.fit(X_train_sca,Y_train)
y_pred=model.predict(X_test_sca)
pro=model.predict_proba(X_test_sca)
acc=accuracy_score(Y_test,y_pred)
print("Accuracy score ",acc)
con=confusion_matrix(Y_test,y_pred)
print("confision matrix ",con)
clas=classification_report(Y_test,y_pred)
print("Classification Report",clas)
# ont of 767 || 268 values are 1(disbetic) and others are nor diabetic 
# So our model is more accurate in the in predicting the non Diabetic patients  
auc=roc_auc_score(Y_test,pro[:,1])
print("Auc Score",auc)

#---------------------------Ruc Curve--------------------
fpr,tpr,thershold=roc_curve(Y_test,pro[:,1])
plt.plot(tpr,fpr,color='red',ls=':',label='Roc curve')
plt.plot([0,1],[0,1],color='black',ls='--',label='Random guess')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate ")
plt.grid(color='grey',linestyle=':',linewidth='0.2')
plt.title("ROC Curve")
plt.legend()
plt.show()
