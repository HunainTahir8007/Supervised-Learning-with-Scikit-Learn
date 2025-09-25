import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

df=pd.read_csv("d:\\cssv\\gym.csv")
#label encoder is only for the one column 

le=LabelEncoder()
X=le.fit_transform(df['Duration'])
y=le.fit_transform(df['Calories'])
print(X)
print(y)

X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.2,test_size=0.8,shuffle=True,random_state=42)
print("x train",X_train)
print("X test",X_test)
print("Y trained",y_train)
print("y test", y_test)