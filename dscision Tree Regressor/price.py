import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from sklearn.datasets import fetch_california_housing
import seaborn as sns
from sklearn.metrics import r2_score,root_mean_squared_error 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree 

data = fetch_california_housing(as_frame=True)
from sklearn.datasets import fetch_california_housing
data = fetch_california_housing(as_frame=True)
df = data.frame.copy()
df['MedHouseVal'] = data.target 

X = df.drop('MedHouseVal', axis=1).values
Y = df['MedHouseVal'].values

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=0.8,random_state=42)

scalar=StandardScaler()
X_test_scaled=scalar.fit_transform(X_test)
X_train_scaled=scalar.transform(X_train)

model=DecisionTreeRegressor(criterion='poisson',max_depth=10,min_samples_split=5,min_samples_leaf=4,random_state=42)
model.fit(X_train_scaled,Y_train)
y_pred=model.predict(X_test_scaled)

r2=r2_score(Y_test,y_pred)
print('R2 score ',r2)
RMSE=root_mean_squared_error(Y_test,y_pred)
print("RMSE",RMSE)

#-----------------plotting the tree-------------------------
# fig,axes=plt.subplots(ncols=1,nrows=1,figsize=(4,4),dpi=200)
# tree.plot_tree(model,filled=True,rounded=True)
# plt.show()
#-----------------------------------------------------------
plt.scatter(Y_test, y_pred, color='black', label='Predicted vs Actual')
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], color='red', linestyle='--', label='Perfect Fit')
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.legend()
plt.show()
