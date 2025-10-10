import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,root_mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

data = {
    "Experience": [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
    "Hours_Studied": [2,5,3,7,4,6,5,8,3,6,7,5,8,4,6,7,5,9,6,8],
    "Teamwork_Score": [55,60,58,65,62,70,68,75,72,78,80,77,83,79,85,88,84,90,87,92],
    "Productivity": [10,25,18,40,30,50,45,65,38,55,60,52,70,58,78,85,72,95,80,105]
}
df=pd.DataFrame(data)
X=df[['Experience' ,  'Teamwork_Score']].values

Y=df['Productivity']
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=0.9,random_state=42)
X_sca=StandardScaler()

X_train_scaled=X_sca.fit_transform(X_train)
X_test_scaled=X_sca.transform(X_test)

mod=SVR(kernel='rbf',C=1000,gamma='scale')
model=mod.fit(X_train_scaled,Y_train)
y_pred=mod.predict(X_test_scaled)
print('r2',r2_score(Y_test,y_pred))
#--------test-----------------
my=np.array([[22,94]])
tr=X_sca.transform(my)
pre=mod.predict(tr)
print(pre)