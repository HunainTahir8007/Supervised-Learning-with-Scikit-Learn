import numpy as np
from sklearn.preprocessing import PolynomialFeatures
X = np.array([[2, 3]])
poly = PolynomialFeatures(degree=2)
print(poly.fit_transform(X))
#degree controls how far the polynomial expansion goes.
