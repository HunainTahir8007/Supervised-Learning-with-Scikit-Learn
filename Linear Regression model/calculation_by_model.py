#in this programm we take the same data and see how the calculation perfrom by the model 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression


Week=[1,2,3,4,5]
Sales=[2,4,5,4,5]

df=pd.DataFrame({"week":Week,'Sales':Sales})
X=df['week'].values
Y=df['Sales'].values
#X is the feture value  (1 D)
#Y is the target value  (1 D)
#Note:-
#X is in the 1 D form we haveto convert it into the 2D form because the model accepts the 2D form 
X=X.reshape((5,1)) #now it change into the 2D form now the model accepts 
#Why reshape only X but not Y?
#Because Y is the target vector, scikit-learn is fine with it being 1
reg=LinearRegression()
reg=reg.fit(X,Y) #we only now fit the data 
r2=reg.score(X,Y)
print("Predicted values:", reg.predict(X))
print(r2)



