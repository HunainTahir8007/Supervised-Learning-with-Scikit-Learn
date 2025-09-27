#one hottencoder creates the binary columns
import numpy as np
from sklearn.preprocessing import OneHotEncoder
data=[['male','female'],['male','female'],['male','female'],['male','female']]
print(np.ndim(data))
on=OneHotEncoder()
readed_data=on.fit_transform(data)
print(readed_data)

#OneHotEncoder works column by column, making new binary columns 
# for each unique category inside each original column.