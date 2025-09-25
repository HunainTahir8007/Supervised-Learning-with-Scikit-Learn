import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
pd.options.display.max_columns=None
data={
    'Names':['Hunain',"Ali",'Ahmad','Abdullah',"Zeeshan"],
    'Age'  :['18','19','17','18','20'],
    'Score':['75','70','85','90','82']
}

read=pd.DataFrame(data).copy()
print("Original dataFrame")
print(read)

ss=LabelEncoder()
X=ss.fit_transform(read['Names'])
y=ss.fit_transform(read['Score'])
print("After Reading by Machine")
print(X)
print(y)
#----------------------Testing and Training the model-----------------------------
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.8,train_size=0.2,shuffle=True,random_state=42)
#0.8 means 80% of data is under the test
#0.2 means the 20 %  of the data is under the train
print(X_test)
print(X_train)
print(y_test)
print(y_train)