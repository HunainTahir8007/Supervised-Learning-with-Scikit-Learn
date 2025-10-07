import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree 
from yellowbrick.model_selection import validation_curve
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score,f1_score,classification_report,confusion_matrix,roc_auc_score,roc_curve

pd.options.display.max_columns=Noneiris = None
iris=load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["target"] = iris.target
X=df.drop('target',axis=1)
Y=df['target'].values

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=0.8,random_state=42)
model=RandomForestClassifier()
model.fit(X_train,Y_train)
y_pred=model.predict(X_test)
print("Accuracy score",accuracy_score(Y_test,y_pred))
print('Confusion Matrix',confusion_matrix(Y_test,y_pred))
print("Classification Matrix",classification_report(Y_test,y_pred))
print("AUC",accuracy_score(Y_test,y_pred))
#------------------------feature that play role in the target---------------------
impor=model.feature_importances_
col=X.columns

plt.barh(col,impor,color='red')
plt.show()
#---------------------------------------------------------------------------
estimator = model.estimators_[0]

# Plot the tree
fig,axes=plt.subplots(nrows=1,ncols=1,figsize=(4,4),dpi=150)
tree.plot_tree(estimator,
          filled=True,
          rounded=True)
plt.show()