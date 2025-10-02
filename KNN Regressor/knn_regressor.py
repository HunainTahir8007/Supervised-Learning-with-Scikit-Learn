import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
import seaborn as sns
from sklearn.preprocessing  import StandardScaler
from sklearn.metrics import r2_score,root_mean_squared_error
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsRegressor
data = {
    "Area": [750, 800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200,
             1250, 1300, 1350, 1400, 1450, 1500, 1600, 1700, 1800, 2000],
    
    "Rooms": [2, 2, 2, 3, 3, 3, 3, 4, 4, 4,
              4, 5, 5, 5, 5, 6, 6, 6, 7, 8],
    
    "Price": [150000, 155000, 160000, 170000, 175000, 180000, 185000, 195000, 200000, 210000,
              220000, 230000, 240000, 250000, 260000, 275000, 290000, 310000, 330000, 360000]
}
 
df=pd.DataFrame(data)
X=df[["Area","Rooms"]].values
Y=df['Price'].values
Y=Y.reshape((-1,1))
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=0.8,random_state=42)

X_sca=StandardScaler()
Y_sca=StandardScaler()

X_test_scaled=X_sca.fit_transform(X_test)
X_train_scaled=X_sca.transform(X_train)

Y_test_scaled=Y_sca.fit_transform(Y_test).ravel()
Y_train_scaled=Y_sca.transform(Y_train).ravel()

model=KNeighborsRegressor(n_neighbors=2)
model.fit(X_train_scaled,Y_train_scaled)
y_pred=model.predict(X_test_scaled)
r2=r2_score(Y_test_scaled,y_pred)
print("r2 :",r2)
rmsv=root_mean_squared_error(Y_test_scaled,y_pred)
print("RMSE",rmsv)

#--------------------testing-----------------------
my=np.array([[850,2]])
tr=X_sca.transform(my)
pre=model.predict(tr).reshape((-1,1))
final=Y_sca.inverse_transform(pre)
print("Pedicted by model",final)
