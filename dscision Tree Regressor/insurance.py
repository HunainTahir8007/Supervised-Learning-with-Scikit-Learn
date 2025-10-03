import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split 
from sklearn.metrics import r2_score,root_mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree


pd.options.display.max_columns=None
df=pd.read_csv('d:\\cssv\\insurance.csv')
X=df.drop('charges',axis=1)
Y=df['charges'].values
from sklearn.tree import _tree
encoder = LabelEncoder()
X['sex'] = encoder.fit_transform(X['sex'])
X['smoker'] = encoder.fit_transform(X['smoker'])
X['region'] = encoder.fit_transform(X['region'])

scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)

X_train,X_test,Y_train,Y_test=train_test_split(X_scaled,Y,train_size=0.8,random_state=42)

model=DecisionTreeRegressor(criterion='absolute_error',max_depth=3,min_samples_split=5,min_samples_leaf=4,random_state=42)
model.fit(X_train,Y_train)
y_pred=model.predict(X_test)
print("r2 score",r2_score(Y_test,y_pred))
print("RMSE",root_mean_squared_error(Y_test,y_pred)) 
#------------------------------testing--------------------------
# Example new data (age, sex, bmi, children, smoker, region)
# Note: must follow the same encoding and scaling as training!
new_data = pd.DataFrame({
    "age": [40],
    "sex": ["male"],
    "bmi": [27.5],
    "children": [2],
    "smoker": ["yes"],
    "region": ["southeast"]
})


new_data['sex'] = encoder.fit(df['sex']).transform(new_data['sex'])
new_data['smoker'] = encoder.fit(df['smoker']).transform(new_data['smoker'])
new_data['region'] = encoder.fit(df['region']).transform(new_data['region'])


new_data_scaled = scaler.transform(new_data)


predicted_charge = model.predict(new_data_scaled)
print("Predicted Insurance Charge:", predicted_charge[0])

#----------------------plotting tree------------------------
fig,axes=plt.subplots(nrows=1,ncols=1,figsize=(4,4),dpi=500)
tree.plot_tree(model)
plt.show()
#----------------------------------------------------------