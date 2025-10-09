import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score,root_mean_squared_error
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVR

data = {
    "Feature1": [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    "Target":   [-60, -35, -18, -5, -2, 0, 2, 5, 20, 38, 65]
}

df=pd.DataFrame(data)
X=df['Feature1'].values
Y=df['Target'].values
X=X.reshape((-1,1))
Y=Y.reshape((-1,1))
X_sca=StandardScaler()
y_sca=StandardScaler()
X_scaled=X_sca.fit_transform(X)
Y_scaled=y_sca.fit_transform(Y).ravel()

model=SVR(kernel='poly',C=1000,gamma='scale',degree=3)
model.fit(X_scaled,Y_scaled)
y_pred=model.predict(X_scaled)
print('R2 score',r2_score(Y_scaled,y_pred))
print("RMSV",root_mean_squared_error(Y_scaled,y_pred))
y_pred=y_pred.reshape((-1,1))
inv=y_sca.inverse_transform(y_pred)
# print(inv)
#---------------------------testing-----------------------
my=np.array([[-6]])
my=X_sca.transform(my)
mod=model.predict(my).reshape((-1,1))
final=y_sca.inverse_transform(mod)
print(final)

#-------------------------plotting -----------------------
X_range=np.linspace(min(X),max(X),500).reshape((-1,1))
X_range_sca=X_sca.transform(X_range)

Y_r=model.predict(X_range_sca).reshape((-1,1))
Y_final=y_sca.inverse_transform(Y_r)

plt.scatter(X,Y,label='Actual points',color='black',edgecolors='grey')
plt.plot(X_range,Y_final,color='red',ls="--",label='Model prediction')
plt.legend(loc='best')
plt.title("Smooth Curve")
plt.show()
