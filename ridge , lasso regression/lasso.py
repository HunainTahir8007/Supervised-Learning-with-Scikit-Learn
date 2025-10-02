import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

np.random.seed(42)
n_samples = 100

X1 = np.random.rand(n_samples) * 10
X2 = X1 + np.random.normal(0, 1, n_samples)   
X3 = np.random.rand(n_samples) * 10
y = 3*X1 + 1.5*X2 - 2*X3 + np.random.normal(0, 2, n_samples)  

df = pd.DataFrame({"X1": X1, "X2": X2, "X3": X3, "Target": y})
X = df[["X1", "X2", "X3"]].values
Y = df["Target"].values


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

lin = LinearRegression()
lin.fit(X_train, Y_train)
y_pred_lin = lin.predict(X_test)

print("Linear Regression R²:", r2_score(Y_test, y_pred_lin))
print("Linear Coefficients:", lin.coef_)

ridge = Ridge(alpha=10) 
ridge.fit(X_train, Y_train)
y_pred_ridge = ridge.predict(X_test)

print("\nRidge Regression R²:", r2_score(Y_test, y_pred_ridge))
print("Ridge Coefficients:", ridge.coef_)

lasso = Lasso(alpha=0.1)
lasso.fit(X_train, Y_train)
y_pred_lasso = lasso.predict(X_test)

print("\nLasso Regression R²:", r2_score(Y_test, y_pred_lasso))
print("Lasso Coefficients:", lasso.coef_)

plt.figure(figsize=(12,5))

plt.subplot(1,3,1)
plt.scatter(Y_test, y_pred_lin, color="blue")
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], "r--")
plt.title("Linear Regression")

plt.subplot(1,3,2)
plt.scatter(Y_test, y_pred_ridge, color="green")
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], "r--")
plt.title("Ridge Regression")

plt.subplot(1,3,3)
plt.scatter(Y_test, y_pred_lasso, color="orange")
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], "r--")
plt.title("Lasso Regression")
plt.show()