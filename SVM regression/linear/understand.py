import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from  sklearn.svm import SVR
from sklearn.metrics  import r2_score
X = np.arange(1, 11).reshape(-1, 1)   # 1 to 10
y = np.array([3, 5, 7, 9, 12, 15, 18, 20, 21, 25])

mod=SVR(kernel='linear')
mod=mod.fit(X,y)
y_pred=mod.predict(X)

print('r2 score ',r2_score(y,y_pred))

plt.scatter(X,y,color='black')
plt.plot(X,y_pred,label='svr predict',color='red',ls='--')
plt.legend()
plt.show()