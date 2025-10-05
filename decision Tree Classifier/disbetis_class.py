#we campare 2 models and plot which model is working best 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,roc_auc_score,roc_curve,f1_score
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import tree

pd.options.display.max_columns=None

df=pd.read_csv('d:\\cssv\\diabetes.csv')
X=df.drop('Outcome',axis=1).values
Y=df['Outcome'].values

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=0.8,random_state=42)
X_sca=StandardScaler()
X_train=X_sca.fit_transform(X_train)
X_test=X_sca.transform(X_test)

#----------------------------logistic regression ---------------------------
lr=LogisticRegression()
lr.fit(X_train,Y_train)
lr_y_pred=lr.predict(X_test)
lr_pro=lr.predict_proba(X_test)
lr_accuracy=accuracy_score(Y_test,lr_y_pred)
print("Logictic Accuracy :",lr_accuracy)
lr_f1=f1_score(Y_test,lr_y_pred)
print("Logistic F1 score ",lr_f1)
lr_con=confusion_matrix(Y_test,lr_y_pred)
print("LogisticConfusion matrix : ",lr_con)
lr_class=classification_report(Y_test,lr_y_pred)
print("Logistic Classification Report",lr_class)
lr_auc=roc_auc_score(Y_test,lr_pro[:,1])
print('logictic AUC',lr_auc)

#-------------------------Desision tree -----------------------------------
tre=DecisionTreeClassifier(criterion='entropy',  max_depth=4,min_samples_split=10,min_samples_leaf=5,random_state=42)
tre.fit(X_train,Y_train)
tre_y_pred=tre.predict(X_test)
tre_pro=tre.predict_proba(X_test)
tre_accuracy=accuracy_score(Y_test,tre_y_pred)
print("Tree Accuracy :",tre_accuracy)
tre_f1=f1_score(Y_test,tre_y_pred)
print("Tree  F1 score ",tre_f1)
tre_con=confusion_matrix(Y_test,tre_y_pred)
print("Tree Confusion matrix : ",tre_con)
tre_class=classification_report(Y_test,tre_y_pred)
print("Tree  Classification Report",tre_class)
tre_auc=roc_auc_score(Y_test,tre_pro[:,1])
print('Tree  AUC',tre_auc)

#----------------------------Roc curves-------------------------------------
lr_fpr,lr_tpr,lr_threshold=roc_curve(Y_test,lr_pro[:,1])
tre_fpr,tre_tpr,tre_threshold=roc_curve(Y_test,tre_pro[:,1])
plt.plot(lr_fpr,lr_tpr,label='Logistic regression',ls=':',color='red')
plt.plot(tre_fpr,tre_tpr,label=' Decission Tree',ls='--',color='blue')
plt.plot([0,1],[0,1],label='Sepreation line',color='black')
plt.legend(loc='best')
plt.title("Decision Tree Vs Logistic Regression")
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.show()

data={
    'Models':["Logictic regression ","Decision Tree"],
   'Accuracy':[lr_accuracy,tre_accuracy],
   'F1_score':[lr_f1,tre_f1],
    'Auc':[lr_auc,tre_auc]
}

final=pd.DataFrame(data)
print(data)