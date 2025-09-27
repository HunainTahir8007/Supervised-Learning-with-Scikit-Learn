import numpy as np
from sklearn.preprocessing import MinMaxScaler

data=np.array([[1,1000],[2,2000],[3,3000]])
mm=MinMaxScaler()
read_data=mm.fit_transform(data)
print(read_data)

#x′=(x-min)/(max-min)