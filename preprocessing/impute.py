#impute meethod is the part of sklearn library in which we can replace the Nan value in the dataset
from  sklearn.impute import SimpleImputer
import numpy as np

arr=np.array([[1,2,3],[4,5,np.nan],[7,np.nan,9]])
imp=SimpleImputer(strategy='mean')# strategies: mean, median, most_frequent, constant
#it takes mean Column by Column NOt row by Row

#Row 1 → [1, 2, 3]
#Row 2 → [4, 5, NaN]
#Row 3 → [7, NaN, 9]


modified_data=imp.fit_transform(arr)
#fit transform meaning:
#Read the data and change it according to it 
print(modified_data)