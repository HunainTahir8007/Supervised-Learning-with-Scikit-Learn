import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.metrics import r2_score,root_mean_squared_error
from sklearn.model_selection import train_test_split,cross_val_score,RandomizedSearchCV
from xgboost import XGBRegressor


pd.options.display.max_columns=None
df=pd.read_csv("d:\\cssv\\car_price.csv")
X=df.drop('price',axis=1)
y=df['price'].values
Y=y.reshape((-1,1))

label=LabelEncoder()
X['fuel']=label.fit_transform(X['fuel'])

X_scalar=StandardScaler()
vals=['year','km_driven','engine','power','seats']
for i in vals:
    X[i]=X_scalar.fit_transform(X[[i]])

X=X.values
Y_scalar=StandardScaler()
Y=Y_scalar.fit_transform(Y).ravel()

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=0.80,random_state=42)

model=XGBRegressor(subsample=0.6,n_estimators=300,min_child_weight=3,max_depth=2,learning_rate=0.3,colsample_bytree=1)
model.fit(X_train,Y_train)
y_pred_test=model.predict(X_test)
y_pred_train=model.predict(X_train)
print("test accuracy",r2_score(Y_test,y_pred_test))
print("train acc",r2_score(Y_train,y_pred_train))
print("RMSE",root_mean_squared_error(Y_test,y_pred_test))
#----------------------------------cross validation scores--------------------
scores = cross_val_score(model, X, Y, cv=5, scoring="r2")
print("CV Scores:", scores)
print("Mean CV Score:", scores.mean())
#--------------------------cheaking the best hyperparmetes to increase the performance----------------------
# data={
#   'n_estimators': [100, 200, 300],
#     'max_depth': [2, 10, 40],
#     'subsample': [0.6, 0.8, 1],
#     'learning_rate': [0.01, 0.1, 0.3],
#     'colsample_bytree': [0.8, 1.0],
#     'min_child_weight': [1, 3, 5],
# }
# cv=RandomizedSearchCV(
#     estimator=XGBRegressor(random_state=42),
#     cv=2,
#     param_distributions=data,
#     scoring='r2',
#     n_iter=30,
#     n_jobs=-1
# )
# cv.fit(X_train,Y_train)

# print("best parameters : ",cv.best_params_)
#-------------------------------------------------------------------------------------