import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
from sklearn.svm import SVR
from sklearn.metrics import r2_score,root_mean_squared_error

data = {
    "Level":[1,2,3,4,5,6,7,8,9,10],
    "Salary":[45000,50000,60000,80000,110000,150000,200000,300000,500000,1000000]
}
df=pd.DataFrame(data)
X=df['Level'].values
Y=df['Salary'].values
X=X.reshape((-1,1))
Y=Y.reshape((-1,1))
Y_log=np.log(Y)

x_sc=StandardScaler()
y_sc=StandardScaler()
X_sca=x_sc.fit_transform(X)
Y_sca=y_sc.fit_transform(Y_log).ravel()

model=SVR(kernel='rbf',gamma=10,C=100)
mod=model.fit(X_sca,Y_sca)
y_pred_log=model.predict(X_sca)
y_pred_log=y_pred_log.reshape((-1,1))
y_pred_log_inv=y_sc.inverse_transform(y_pred_log)
y_pred_final=np.exp(y_pred_log_inv)
# print('y pred',y_pred_final)
r2=r2_score(Y,y_pred_final)
print("r2 ",r2)
rmsv=root_mean_squared_error(Y,y_pred_final)
print('rmse',rmsv)
#----------------------testing----------------------------
arr=np.array([[12]])
arr_sca=x_sc.transform(arr)
pred=model.predict(arr_sca).reshape((-1,1))
pred_inv=y_sc.inverse_transform(pred)
final_prediction=np.exp(pred_inv)
print('TESTING ::',final_prediction)
#-----------------------plotting --------------------------
X_range=np.linspace(max(X),min(X),300)
X_range=X_range.reshape((-1,1))
X_range_scaled=x_sc.transform(X_range)

Y_range_sca=model.predict(X_range_scaled)
Y_range_sca=Y_range_sca.reshape((-1,1))
Y_range_log=y_sc.inverse_transform(Y_range_sca)
Y_range_final=np.exp(Y_range_log)

plt.scatter(X,Y,color='black',label='Actual Points')
plt.plot(X_range,Y_range_final,color='red',ls='--',label="Model predict")
plt.title('Smooth curve')
plt.title("SVR with Log Salary Transformation")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.grid(color='grey', linewidth=0.3, linestyle=":")
plt.legend(loc='best')
# plt.show()