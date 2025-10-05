import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,roc_auc_score,roc_curve
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

df=pd.read_csv("d:\\cssv\\play.csv")
X=df.drop('play',axis=1)
Y=df['play'].values
X_sca=LabelEncoder()
X_scaled=X.apply(LabelEncoder().fit_transform).values

Y_sca=LabelEncoder()
Y_scaled=Y_sca.fit_transform(Y)
X_train,X_test,Y_train,Y_test=train_test_split(X_scaled,Y_scaled,train_size=0.8,random_state=42)
model=DecisionTreeClassifier(criterion='entropy',splitter='best')
model.fit(X_train,Y_train)
y_pred=model.predict(X_test)
pro=model.predict_proba(X_test)
print("Probability",pro)
acc=accuracy_score(Y_test,y_pred)
print("Accuracy ",acc)
con=confusion_matrix(Y_test,y_pred)
print("Confussion Matrix",con)
clas=classification_report(Y_test,y_pred)
print("Classification Report",clas)
Auc=roc_auc_score(Y_test,pro[:,1])
print("AUC",Auc)
#---------------------------ROC-------------------------------
fpr,tpr,threshold=roc_curve(Y_test,pro[:,1])
plt.plot(fpr,tpr,label='Roc Curve',color='red',ls=":")
plt.plot([0,1],[0,1],color='black',label='Sepreation line',ls='--')
plt.title("ROC Curve Movement")
plt.legend(loc='best')
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.show()

#------------------------Plotting the tree-----------------------------------
fig,axes=plt.subplots(nrows=1,ncols=1,figsize=(4,4),dpi=150)
tree.plot_tree(model,feature_names=X.columns, 
               class_names=Y_sca.classes_, 
               filled=True, 
               rounded=True,
               fontsize=6,
               ax=axes)
plt.show()

