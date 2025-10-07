import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import StandardScaler,LabelEncoder,OrdinalEncoder
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree 
from yellowbrick.model_selection import validation_curve
from sklearn.metrics import accuracy_score,f1_score,classification_report,confusion_matrix,roc_auc_score,roc_curve,auc
from sklearn.preprocessing import label_binarize
pd.options.display.max_columns=None

df=pd.read_csv('F:\\cssv\\dermatology.csv', na_values="?")
df.dropna(inplace=True)
X=df.drop('class',axis=1).values
Y=df['class'].values
sca=StandardScaler()
x_scaled=sca.fit_transform(X)

X_train,X_test,Y_train,Y_test=train_test_split(x_scaled,Y,train_size=0.8,random_state=42)
model=RandomForestClassifier(criterion='entropy')
model.fit(X_train,Y_train)
y_pred=model.predict(X_test)
pro=model.predict_proba(X_test)
print("Accuracy score",accuracy_score(Y_test,y_pred))
print('Confusion Matrix',confusion_matrix(Y_test,y_pred))
print("Classification Matrix",classification_report(Y_test,y_pred))
print("AUC",accuracy_score(Y_test,y_pred))
#----------------------------ROC Curve---------------------
Y_test_binar=label_binarize(Y_test,classes=[0,1,2,3,4,5])
fpr,tpr,threshold=roc_curve(Y_test_binar.ravel(),pro.ravel())
# Loop for each class
for i in range(Y_test_binar.shape[1]):
    fpr, tpr, _ = roc_curve(Y_test_binar[:, i], pro[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'Class {i+1} (AUC={roc_auc:.2f})')

plt.plot([0,1],[0,1],'k--',label='Separation line')
plt.title("ROC Curve per Class")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()
#------------------------------plotting tree (10)-----------------------
for i in range(1,10):
    estim=model.estimators_[i]
    fig,axes=plt.subplots(ncols=1,nrows=1,dpi=200)
    tree.plot_tree(estim,filled=True,rounded=True)
    plt.show()
    