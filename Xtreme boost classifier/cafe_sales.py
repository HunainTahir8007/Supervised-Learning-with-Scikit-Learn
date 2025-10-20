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
df=pd.read_csv("F:\\cssv\\cafe_sales.csv")
df.replace('UNKNOWN',np.nan,inplace=True)
df.replace('ERROR',np.nan,inplace=True)
df.dropna(inplace=True)
df.reset_index(inplace=True)
print(df)
df.drop('Transaction Date',axis=1,inplace=True)
df.drop('Transaction ID',axis=1,inplace=True)
df.drop('index',axis=1,inplace=True)
df.drop('Payment Method',axis=1,inplace=True)
df.drop('Location',axis=1,inplace=True)

#---------------------Assigning the value----------------------
x=df.drop('Item',axis=1)
Y=df['Item'].values


sca = ['Quantity','Price Per Unit','Total Spent']
X_scaler=StandardScaler()
for i in sca:
    x[i]=X_scaler.fit_transform(x[[i]])

X=x.values

Y_encoder=LabelEncoder()
Y=Y_encoder.fit_transform(Y)

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=0.8,random_state=42)

model=XGBClassifier()
model.fit(X_train,Y_train)
y_pred_test=model.predict(X_test)
y_pred_train=model.predict(X_train)
print("Test Accuray",accuracy_score(Y_test,y_pred_test))
print("Train Accuray",accuracy_score(Y_train,y_pred_train))

#--------------------cross validation score-------------------------
scores = cross_val_score(model, X, Y, cv=5, scoring='accuracy')
print(scores)
print("CV Mean:", scores.mean())
#----------------Cheaking the feature importances -----------------
imp=model.feature_importances_
col=x.columns
zz=plt.bar(col,imp,color='red')
plt.bar_label(zz)
plt.show()
#-------------cheaking the best parameters-----------------------
from sklearn.model_selection import GridSearchCV
params = {
    'max_depth': [3, 5, 7],
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0]
}

grid = GridSearchCV(
    estimator=XGBClassifier(random_state=42),
    param_grid=params,
    scoring='accuracy',
    cv=3,
    verbose=1,
    n_jobs=-1
)

grid.fit(X_train,Y_train )
print("Best Params:", grid.best_params_)
print("Best CV Score:", grid.best_score_)
#Connclusion:
   # the model has strong feature is price per unit while others are very waek 
   # By default the model is perfect not use the hyperparameters