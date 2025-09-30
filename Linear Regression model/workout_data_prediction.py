import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.linear_model import LinearRegression

pd.options.display.max_columns=None
df=pd.read_csv('d:\\cssv\\workout.csv')
# in this we will predict the height by the age and weight 
X = df[['Duration','Pulse','Weight']]
Y = df['Calories'].values

# print(df.corr()['Calories']) # this tells that the colums that strogly related to the calories

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=0.8,random_state=42)
model=LinearRegression()
model=model.fit(X_train,Y_train)
y_pred=model.predict(X_test)
# r2=r2_score(Y_test,y_pred)
# print(r2)
# rms=mean_squared_error(Y_test,y_pred)
# print(rms)
#-------------------making the prediction --------------
# print(X,Y)
# print(model.predict([[44,121,93]]))
#train and test can take the random values so we sort itX_sorted = np.sort(X_duration.values, axis=0)



#---------------ploting------------
# Plot Duration vs Calories
plt.scatter(X['Duration'], Y, color="blue", label="Actual")

# Generate regression line (keeping Pulse & Weight fixed at mean)
duration_range = np.linspace(X['Duration'].min(), X['Duration'].max(), 100)
pulse_mean = X['Pulse'].mean()
weight_mean = X['Weight'].mean()

X_line = np.column_stack((duration_range, 
                          np.full(100, pulse_mean), 
                          np.full(100, weight_mean)))
y_line = model.predict(X_line)

# Plot regression line
plt.plot(duration_range, y_line, color="red", linewidth=2, label="Regression Line")

plt.xlabel("Duration")
plt.ylabel("Calories")
plt.legend()
plt.show()
