import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import StandardScaler,LabelEncoder,OrdinalEncoder
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree 
from yellowbrick.model_selection import validation_curve
from sklearn.metrics import accuracy_score,f1_score,classification_report,confusion_matrix,roc_auc_score,roc_curve
from sklearn.preprocessing import label_binarize

pd.options.display.max_columns=Noneiris = None
df=pd.read_csv('F:\\cssv\\titanic.csv')
df.dropna(inplace=True)
df=df.drop('Ticket',axis=1)
df=df.drop('PassengerId',axis=1)
df=df.drop('Cabin',axis=1)
df=df.drop('Name',axis=1)
X=df.drop('Embarked',axis=1)
Y=df['Embarked'].values

scalar=StandardScaler()

X['Age']=scalar.fit_transform(X[['Age']])
X['Fare']=scalar.fit_transform(X[['Fare']])

label=LabelEncoder()
X['Sex']=label.fit_transform(X['Sex'])
Y=label.fit_transform(Y)
cols=X.columns
X=X.values

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=0.8,random_state=42)

model=RandomForestClassifier(criterion='gini',random_state=42)
model.fit(X_train,Y_train)
y_pred=model.predict(X_test)
pro=model.predict_proba(X_test)
print("Accuracy score",accuracy_score(Y_test,y_pred))
print('Confusion Matrix',confusion_matrix(Y_test,y_pred))
print("Classification Matrix",classification_report(Y_test,y_pred))
print("AUC",accuracy_score(Y_test,y_pred))
#--------------------Cheaking the contribution of eaach Feature------------------
# importances=model.feature_importances_
# col=cols
# fx=plt.barh(col,cols,color='red',label='Contribution of each feature')
# plt.legend(loc='best')
# plt.bar_label(fx)
# plt.show()
#Result:- passinger id cannot play role in the output so we remove it 
# --------------------------------------------------------
#--------------------Cheaking the best parameters----------------------
est_test=[2,4,7,10,15,18,20,22]
# print(validation_curve(RandomForestClassifier(),X_train,Y_train,param_name='n_estimators',param_range=est_test,cv=3,scoring='accuracy'))
#------------------------------------------------------
dep=[2,4,6,8,10,13,14]
# print(validation_curve(RandomForestClassifier(),X_train,Y_train,param_name='max_depth',param_range=dep,cv=3,scoring='accuracy'))
min_samp=[2,3,5,7,9,10,12,14,15]
# print(validation_curve(RandomForestClassifier(),X_train,Y_train,param_name='min_samples_split',param_range=min_samp,cv=3,scoring='accuracy'))
#----------------------------------------------------------------------
#-------------------------ROC Curve--------------------------
Y_test_bin = label_binarize(Y_test, classes=[0,1,2])

fpr, tpr, _ = roc_curve(Y_test_bin.ravel(), pro.ravel())
plt.plot(fpr,tpr,label='Roc Curve',color='red',ls=":")
plt.plot([0,1],[0,1],color='black',label='Sepreation line',ls='--')
plt.title("ROC Curve Movement")
plt.legend(loc='best')
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.show()
#------------------------------------------------------------
#------------------------plotting the trees -------------------------
# estim=model.estimators_[0]
# fig,axes=plt.subplots(ncols=1,nrows=1,dpi=200)
# tree.plot_tree(estim,filled=True,rounded=True)
# plt.show()