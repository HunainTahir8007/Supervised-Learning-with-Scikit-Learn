import re
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import seaborn as sns 
from sklearn.metrics import r2_score,root_mean_squared_error
from sklearn.preprocessing import LabelEncoder,StandardScaler,OrdinalEncoder
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV
from xgboost import XGBRegressor
from sklearn import tree
from xgboost import plot_tree


pd.options.display.max_columns=None
df=pd.read_csv('F:\\cssv\\laptop.csv')
df.drop('Unnamed: 0',axis=1,inplace=True)
df.dropna(inplace=True)
#----------------modification of the resulution-----------------------------
df['Xresolution']=df['ScreenResolution'].str.extract(r'(\d+)x')[0].astype(float)
df['Yresolution']=df['ScreenResolution'].str.extract(r'x(\d+)')[0].astype(float)

df['Aspect resolution']=(df['Xresolution'] / df['Yresolution']).round(2)

#---------------------------Chaniging the screenResulution strings to the int------------------------
df['IPS']=df['ScreenResolution'].str.contains('IPS').astype(int)
df['Full HD']=df['ScreenResolution'].str.contains('Full HD').astype(int)
df['Touchscreen']=df['ScreenResolution'].str.contains('Touchscreen').astype(int)
df['Retina'] = df['ScreenResolution'].str.contains('Retina').astype(int)
df['4K'] = df['ScreenResolution'].str.contains('4K').astype(int)
df["QuadHD"] = df["ScreenResolution"].str.contains("Quad HD").astype(int)
df["Retina"] = df["ScreenResolution"].str.contains("Retina").astype(int)

df.drop('ScreenResolution',axis=1,inplace=True)
df.drop(['Xresolution','Yresolution'], axis=1, inplace=True)
#-------------------------------Changing in the CPU--------------------
df['CPU_Brand']=df['Cpu'].apply(lambda x: x.split()[0])
df['CPU_series']=df['Cpu'].apply(lambda x:" ".join(x.split()[1:3]))
df['Clock_speed']=df['Cpu'].apply(lambda x:float(x.split()[-1].replace("GHz","")))
df.drop('Cpu',axis=1,inplace=True)

#-----------------------Changing in Ram---------------------
df['RAM']=df['Ram'].str.replace('GB',"")
df.drop('Ram',axis=1,inplace=True)



# ------------------ Modifying the Memory ------------------


df.reset_index(drop=True, inplace=True)


df["HDD"] = 0
df["SSD"] = 0
df["Hybrid"] = 0
df["Flash"] = 0

def Convert_to_GB(value):
    value = value.upper().replace(" ", "")
    if "TB" in value:
        return float(value.replace("TB", "")) * 1000
    if "GB" in value:
        return float(value.replace("GB", ""))
    else:
     return 0


for i, entry in enumerate(df["Memory"]):
    if entry == "?" or entry.strip() == "":
        continue 

    parts = entry.split("+")  
    for part in parts:
        part = part.strip().upper()  

        if "HDD" in part:
            size = part.replace("HDD", "")
            df.at[i, "HDD"] += Convert_to_GB(size)
        elif "SSD" in part:
            size = part.replace("SSD", "")
            df.at[i, "SSD"] += Convert_to_GB(size)
        elif "HYBRID" in part:
            size = part.replace("HYBRID", "")
            df.at[i, "Hybrid"] += Convert_to_GB(size)
        elif "FLASH" in part:
            size = part.replace("FLASH", "").replace("STORAGE", "")
            df.at[i, "Flash"] += Convert_to_GB(size)

df.drop("Memory", axis=1, inplace=True)
#------------------------------changing in the GPU-----------------
df["GPU_Brand"]=df['Gpu'].apply(lambda x:x.split()[0])
df["GPU_Series"] = df["Gpu"].apply(lambda x: " ".join(x.split()[1:])) 
df.drop('Gpu',inplace=True,axis=1)
df["Type"]=df['TypeName'].str.replace("2 in 1",'')
df.drop('TypeName',axis=1,inplace=True)
df["weight"]=df["Weight"].str.replace('kg','')
df.drop("Weight",axis=1,inplace=True)
#---------------------removing the invalid values-----------------
df.drop('IPS',inplace=True,axis=1)
df.drop('QuadHD',inplace=True,axis=1)
df.drop('Hybrid',inplace=True,axis=1)
df.replace("?", np.nan, inplace=True)
df.replace("", np.nan, inplace=True)
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)
#--------------------Assigning Values------------------------------
x=df.drop('Price',axis=1)
y=df['Price'].values
Y=np.log(y)
#-----------------------------Encode Values------------------------

vals=['Company','OpSys','CPU_Brand','CPU_series','GPU_Brand','GPU_Series','Type']
encoders = {} 
le=LabelEncoder()
for i in vals:
   x[i]=le.fit_transform(x[i])
   encoders[i]=le
X=x.values
#-----------------------Training testing data------------------
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=0.8,random_state=42)

#---------------------scaling the values-------------------------
X_scalar=StandardScaler()
X_train_scaled=X_scalar.fit_transform(X_train)
X_test_scaled=X_scalar.transform(X_test)


model=XGBRegressor(n_estimators=300, learning_rate=0.1, max_depth=8, random_state=42)
model.fit(X_train_scaled,Y_train)
y_pred_test=model.predict(X_test_scaled)
y_pred_train=model.predict(X_train_scaled)
print("r2_test",r2_score(Y_test,y_pred_test))
print("r2_train",r2_score(Y_train,y_pred_train))
print(df['Company'].value_counts())
#----------------------Cross Validation--------------------------------------------
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, Y, cv=5, scoring='r2')
print(scores)
print("Mean R²:", scores.mean())
#----------------------------Testing the model------------------------------------------
testing_data={
    "Company": ["Apple",'Lenovo'],
    'Inches':[13.3,14],
    "OpSys": ["macOS",'Windows 10'],
    "Aspect resolution": [1.60,1.78],
    "Full HD": [0,1],
    "Touchscreen": [0,1],
    "Retina": [1,0],
    "4K": [0,0],
    "CPU_Brand": ["Intel",'Intel'],
    "CPU_series": ["Core i5",'Core i7'],
    "Clock_speed": [2.3,2.5],
    "RAM": [8,4],
    "HDD": [0,0],
    "SSD": [128,128],
    "Flash": [0,0],
    "GPU_Brand": ["Intel",'Intel'],
    "GPU_Series": ["Iris Plus Graphics 640",'HD Graphics 520'],
    "Type": ["Ultrabook",'Convertible'],
    "weight": [1.37,1.8]
   
}
test_frame=pd.DataFrame(testing_data)
my_X_test = test_frame.copy()

test_vals=['Company','OpSys','CPU_Brand','CPU_series','GPU_Brand','GPU_Series','Type']
for i in test_vals:
   le=encoders[i]
   my_X_test[i] = my_X_test[i].map(lambda s: s if s in le.classes_ else le.classes_[0])
   my_X_test[i] = le.transform(my_X_test[i])
   # Scale features
my_X_test_scaled = X_scalar.transform(my_X_test)

pred_log = model.predict(my_X_test_scaled)
pred_prices = np.exp(pred_log)   
print("Predicted Prices:", pred_prices)
# ----------------------------Cheaking the best parameters--------------------------------
data={
   'n_estimators': [100, 200, 300],
    'subsample': [0.7, 0.8, 1.0],
    'learning_rate': [0.01, 0.05, 0.1],
    'objective': ['reg:squarederror'], 
    'colsample_bytree': [0.7, 0.8, 1.0],
    'min_child_weight': [1, 3, 5],
    'max_depth': [3, 5, 8, 10]
}
cv=GridSearchCV(
   estimator=XGBRegressor(random_state=42),param_grid=data,cv=3,n_jobs=-2,scoring='r2'
)
cv.fit(X_train_scaled,Y_train)
print("Best parameters :",cv.best_params_)
print("Best Score : ",cv.best_score_)
#-----------------------------Cheaking the each feature importance------------------------
imp=model.feature_importances_
col=x.columns 
plt.bar(col,imp,color='red',label='Fetaure Importance')
plt.xlabel("Features Names")
plt.ylabel("Feature Importaces")
plt.legend(loc='best')
plt.show()

#--------------------------plotting the model--------------------------------------
plt.scatter(Y_test,y_pred_test,label="Tested Vs Predictted",color='black')
plt.plot([Y_test.min(),Y_test.max()],[Y_test.min(),Y_test.max()],ls=':',color='red')
plt.title("Model Prediction")
plt.grid(ls=":",linewidth=0.3,color='grey')
plt.legend(loc='best')
plt.show()
#-----------------------plotting the tree----------------------------------------
for i in range(4):
    plt.figure(figsize=(15,10), dpi=150)
    plot_tree(model, num_trees=i, rankdir='LR')  
    plt.title(f"Tree number {i}")
    plt.show()
# the plotting of the xgboost Model use the graphviz path 


#Conclusion :-
    # The model is predicting very Good There is no overfitting
    # In testing it gives the difference of 15K in Apple Laptop because the dataset is not balanced (Low Apple laptop Ratio :21)
    #So in case of Lenovo laptop the Prediction is only the difference of 3K (More Lenove laptop Ration)
#-------------------------------------------------------------------------------------------------------------------------