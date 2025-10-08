import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import seaborn as sns 
from sklearn.model_selection import train_test_split,GridSearchCV
from yellowbrick.model_selection import validation_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score,root_mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn import tree

df=pd.read_csv('d:\\cssv\\randomforest.csv')
df.drop("Feature_2",axis=1,inplace=True)
df.drop('Feature_5',axis=1,inplace=True)
X=df.drop('Target',axis=1)
Y=df['Target'].values

scalar=StandardScaler()
X_scaled=scalar.fit_transform(X)

X_train,X_test,Y_train,Y_test=train_test_split(X_scaled,Y,train_size=0.8,random_state=42)
model=RandomForestRegressor(criterion='absolute_error',min_samples_split=10,n_estimators=300,min_samples_leaf=4,max_depth=20)
model.fit(X_train,Y_train)
y_pred=model.predict(X_test)
print("r2 score ",r2_score(Y_test,y_pred))
print("RMSE",root_mean_squared_error(Y_test,y_pred))

#-----------------------cheaking the best hyperparameters-----------------
# param_grid = {
#     'n_estimators': [100, 200, 300],
#     'max_depth': [10, 15, 20],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],
#     'criterion': ['squared_error', 'absolute_error']
# }

# grid_search = GridSearchCV(
#     estimator=RandomForestRegressor(random_state=42),
#     param_grid=param_grid,
#     cv=3,
#     scoring='r2',
#     n_jobs=-1,  
#     verbose=2
# )

# grid_search.fit(X_train, Y_train)

# print("Best Parameters:", grid_search.best_params_)
#----------------------------plotting the trees-----------------------
# for i in range(0,5):
#     estim=model.estimators_[i]
#     fig,axes=plt.subplots(ncols=1,nrows=1,dpi=200)
#     tree.plot_tree(estim,rounded=True,filled=True)
    # plt.show()
#-------------------------plotting---------------------
plt.scatter(Y_test, y_pred, color='black', label='Predicted vs Actual')
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], color='red', linestyle='--', label='Perfect Fit')
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.legend()
plt.show()