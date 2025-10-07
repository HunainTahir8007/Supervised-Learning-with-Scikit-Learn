import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import seaborn as sns
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,roc_auc_score,roc_curve,f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from yellowbrick.model_selection import validation_curve
from sklearn.tree import plot_tree
df=pd.read_csv("d:\\cssv\\students_big.csv")
X=df.drop('Pass',axis=1)
X=X.values
Y=df['Pass'].values
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=0.8,random_state=42)

model=RandomForestClassifier()#it takes parameters now it is default
model.fit(X_train,Y_train)
y_pred=model.predict(X_test)
print("Accuracy score",accuracy_score(Y_test,y_pred))
print("F1 score",f1_score(Y_test,y_pred))
print("Confusion Matrix",confusion_matrix(Y_test,y_pred))
print("Classification Report",classification_report(Y_test,y_pred))
print("AUC",roc_auc_score(Y_test,y_pred))

#---------cheaking the best n_estimator values used in the model----------------------

best_n_est_vals=[100,200,450,500,750,1000]
# print(validation_curve(RandomForestClassifier(),X_train,Y_train,param_name="n_estimators",param_range=best_n_est_vals,scoring='accuracy',cv=3))
#----------------cheaking the best value for the max depth---------------------------
best_max_depth=[5,6,7,8,10,12,15,20]
# print(validation_curve(RandomForestClassifier(),X_train,Y_train,param_name="max_depth",param_range=best_max_depth,scoring='accuracy',cv=3))
#----------------best values for max_sample_split ------------------------------
best_mix_split=[1,3,4,5,7,10,14,15]
# print(validation_curve(RandomForestClassifier(),X_train,Y_train,param_name="min_samples_split",param_range=best_mix_split,scoring='accuracy',cv=3))

#--------------Now predicting with these parameters---------------------------
model2=RandomForestClassifier(criterion='entropy',n_estimators=100,max_depth=7,min_samples_split=3,random_state=42)
model2.fit(X_train,Y_train)
y_pred2=model2.predict(X_test)
print("Accuracy score2",accuracy_score(Y_test,y_pred2))
print("F1 score2",f1_score(Y_test,y_pred2))
print("Confusion Matrix2",confusion_matrix(Y_test,y_pred2))
print("Classification Report2",classification_report(Y_test,y_pred2))
print("AUC2",roc_auc_score(Y_test,y_pred2))

#--------------------------Contribution of each Feature----------------------------------
importances = model2.feature_importances_
features = df.drop('Pass', axis=1).columns

sns.barplot(x=importances, y=features)
plt.title("Feature Importance in Random Forest")
plt.show()
#-----------------------Plotting the 1 tree---------------------------------
estimator = model.estimators_[0]

# Plot the tree
fig,axes=plt.subplots(nrows=1,ncols=1,figsize=(4,4),dpi=150)
plot_tree(estimator,
          filled=True,
          rounded=True)
plt.show()
#-------------------------------------------------------------------------