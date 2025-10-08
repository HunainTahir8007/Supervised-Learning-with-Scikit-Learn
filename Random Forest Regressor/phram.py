import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import seaborn as sns 
from sklearn.model_selection import train_test_split,GridSearchCV
from yellowbrick.model_selection import validation_curve
from sklearn.preprocessing import StandardScaler,OrdinalEncoder
from sklearn.metrics import r2_score,root_mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn import tree
pd.options.display.max_columns=None
df=pd.read_csv("F:\\cssv\\pharm.csv")
print(df)
df.drop('Date',axis=1,inplace=True)
X=df.drop('Amount ($)',axis=1)

Y=df['Amount ($)'].values

ordinal=OrdinalEncoder()
X['Product']=ordinal.fit_transform(X[['Product']])
X['Sales Person']=ordinal.fit_transform(X[['Sales Person']])
X['Country']=ordinal.fit_transform(X[['Country']])

scalar=StandardScaler()
X['Boxes Shipped']=scalar.fit_transform(X[['Boxes Shipped']])

X=X.values

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=0.8,random_state=42)

model=RandomForestRegressor(criterion='squared_error',random_state=42)
model.fit(X_train,Y_train)
y_pred=model.predict(X_test)

print('R2',r2_score(Y_test,y_pred))
print("RMSV",root_mean_squared_error(Y_test,y_pred))

#-----------------------cheaking the best Hyperparameters-----------------
# vals={
# 'n_estimators': [100, 200, 300],
#     'max_depth': [10, 15, 20],
#     'min_samples_split': [2, 5, 10],
#    'min_samples_leaf': [1, 2, 4],
#     'criterion': ['squared_error', 'absolute_error']
# }
# grid_search=GridSearchCV(
#     estimator=RandomForestRegressor(random_state=42),param_grid=vals,cv=3,scoring='r2',n_jobs=-1
# )
# grid_search.fit(X_train,Y_train)
# print("Best Parameters:", grid_search.best_params_)
#-------------------------------------------------------------------------------------------
#-------------------plotting the tree---------------
# for i in range(0,5):
#     estim=model.estimators_[i]
#     fig,axes=plt.subplots(ncols=1,nrows=1,dpi=200)
#     tree.plot_tree(estim,rounded=True,filled=True)
    # plt.show()
#-----------------------------------------------------
plt.scatter(Y_test,y_pred,label='predictions')
plt.plot([Y_test.min(),Y_test.max()],[Y_test.min(),Y_test.max()],label='line')
plt.legend(loc='best')
plt.show()
