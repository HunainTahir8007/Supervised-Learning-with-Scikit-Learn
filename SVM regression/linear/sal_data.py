import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler


df=pd.read_csv('d:\\cssv\\Salary.csv')
X=df['YearsExperience']
X=np.reshape(X,[-1,1])
Y=df['Salary']

scal_X=StandardScaler()

X_sca=scal_X.fit_transform(X)


mod=SVR(kernel='linear',C=1000) #linear reg finds the best line
# SvR form the linear line just if we add C =1000 it tries to passing almost every point
 
mod=mod.fit(X,Y)
y_pred=mod.predict(X)
print('r2 score',r2_score(Y,y_pred))
print("mean _sq",mean_squared_error(Y,y_pred))
#plotting 
plt.scatter(X,Y,color='black',label='Actual values ')
plt.plot(X,y_pred,color='red',label='Model predicted',ls='--')
plt.legend()
plt.show()
