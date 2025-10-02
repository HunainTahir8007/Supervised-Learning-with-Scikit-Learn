import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# ---------------- Dataset in dictionary form ----------------
data = {
    "Feature1": [0.1, 0.2, 0.3, 0.5, 0.6, 0.9, 0.2, 0.8, 0.4, 0.7],
    "Feature2": [0.2, 0.4, 0.1, 0.7, 0.8, 0.4, 0.9, 0.5, 0.6, 0.3],
    "Feature3": [0.3, 0.5, 0.6, 0.9, 0.2, 0.3, 0.8, 0.7, 0.5, 0.4],
    "Target":   [5.1, 7.2, 4.8, 10.5, 9.4, 8.8, 11.3, 12.2, 9.1, 8.4]
}
df=pd.DataFrame(data)
X=df.drop('Target',axis=1).values
Y=df['Target'].values

model=LinearRegression()
model.fit(X,Y)

lin_pre=model.predict(X)
print("R2",r2_score(Y,lin_pre))
#-------------------------ridge----------------------
ride=Ridge(alpha=1)
ride.fit(X,Y)
rid_pred=ride.predict(X)
print("R2 Ridge",r2_score(Y,rid_pred))
#--------------------lesso------------------
les=Lasso(alpha=0.1)
les.fit(X,Y)
les_pred=les.predict(X)
print("r2 les",r2_score(Y,les_pred))
