# scalar is used when we have an array like [1,1000]
#one value is so small and other value is very large so we use the standradscalar method 

import numpy as np
from sklearn.preprocessing import StandardScaler

dat=np.array([[0,1000],[1,2000],[2,3000],[3,4000]])
ss=StandardScaler()
readed_data=ss.fit_transform(dat)
print(readed_data)

#here it applies the formula  z= x-mean/std
# x for original value 
# mean ----
#std = standard deviation 
