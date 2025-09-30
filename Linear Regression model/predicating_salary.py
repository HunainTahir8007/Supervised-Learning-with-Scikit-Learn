import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
df=pd.read_csv('d:\\cssv\\Salary.csv')
X=df['YearsExperience'].values
X=X.reshape((-1,1))
Y=df['Salary'].values
X_scaled=StandardScaler()

X_sca=X_scaled.fit_transform(X)



#data is small no need to train and test 
model=LinearRegression()
model=model.fit(X_sca,Y)
y_pred=model.predict(X_sca)
#now the X is scaled and y is not scaled so if we want to predict value we have to tranfom into rhe scalar
val=np.array([[1.4]])
print(np.ndim(val))
new=X_scaled.transform(val)
print(model.predict(new))

r2=r2_score(Y,y_pred)
print('r2',r2)

rmse = mean_squared_error(Y, y_pred)  # squared=False means RMSE
print("RMSE:", rmse)

#-----------------plotting -----------------------
plt.scatter(X, Y, color='black', label='Actual values')   # Actual data points
plt.plot(X, y_pred, color='red', label='Regression line',ls='--') # Regression line
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Linear Regression: Salary vs Experience")
plt.legend()
plt.show()
