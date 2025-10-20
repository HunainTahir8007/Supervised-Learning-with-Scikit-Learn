import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np 
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,roc_auc_score,roc_curve,f1_score
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score
from xgboost import XGBClassifier
from xgboost import plot_tree

pd.options.display.max_columns=None
df=pd.read_csv("d:\\cssv\\mushrooms.csv")

#------------------------Assigning the values-----------------------
X=df.drop("class",axis=1)
Y=df['class'].values
#-----------------------Encoding the values-------------------------
vals=['cap-shape','cap-surface','cap-color','bruises','odor','gill-attachment','gill-spacing','gill-size','gill-color','stalk-shape','stalk-root','stalk-surface-above-ring','stalk-surface-below-ring','stalk-color-above-ring','stalk-color-below-ring','veil-type','veil-color','ring-number','ring-type','spore-print-color','population','habitat']
X_encoder=LabelEncoder()
for i in vals:
 X[i]=X_encoder.fit_transform(X[i])
X=X.values
Y_encoder=LabelEncoder()
Y=Y_encoder.fit_transform(Y)

#-----------------------Splitting the data--------------------------
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=0.8,random_state=42,shuffle=True)

#---------------------Training the Model---------------------------
model=XGBClassifier()
model.fit(X_train,Y_train)
y_pred_test=model.predict(X_test)
y_pred_train=model.predict(X_train)
#---------------Accuracy Scores to cheak the overfitting----------
print("Accuracy Test",accuracy_score(Y_test,y_pred_test))
print("Accuract Train",accuracy_score(Y_train,y_pred_train))
#---------------------Validation Scores -------------------------
score=cross_val_score(model,X,Y,cv=5,scoring='accuracy')
print(score)
print(score.mean())
#---------------------------Metrices----------------------------
print("Classification Report",classification_report(Y_test,y_pred_test))
print("Confusion Matrix",confusion_matrix(Y_test,y_pred_test))
print("f1 score",f1_score(Y_test,y_pred_test))
print("AUC Score",roc_auc_score(Y_test,y_pred_test))

#Conclusion :--
       # this data is cleaned so this data has accuracy scre is 100%
       # But in real world the data is Messay,and donot give the 100% accuracy
       # so this model is accurate on this due to data is cleaned and balanced