import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

pd.options.display.max_columns=None
df=pd.read_csv("d:\\cssv\\Housing.csv")
Y=df['price'].values.reshape((-1,1))
X=df.drop('price',axis=1)

# ---------------- Scaling numeric features ----------------
X_scalar=StandardScaler()
for col in ['area','bedrooms','bathrooms','stories','parking']:
    X[col]=X_scalar.fit_transform(X[[col]])

# ---------------- Scaling target ----------------
Y_scalar=StandardScaler()
Y=Y_scalar.fit_transform(Y).ravel()

# ---------------- Encoding categorical features ----------------
X_encoder=LabelEncoder()
for col in ['mainroad','guestroom','basement','hotwaterheating',
            'airconditioning','prefarea','furnishingstatus']:
    X[col]=X_encoder.fit_transform(X[col])

X=X.values

# ---------------- Train / Test split ----------------
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=0.8,random_state=42)

# ---------------- Linear Regression ----------------
model=LinearRegression()
model.fit(X_train,Y_train)

# Predictions
y_train_pred=model.predict(X_train)
y_test_pred=model.predict(X_test)

# R² scores
print("Train R²:", r2_score(Y_train,y_train_pred))
print("Test R²:", r2_score(Y_test,y_test_pred))
#there is no difference so the model is not overfitting so if there si a big differcence then we use the ridge and lasso for performmance
