# encode means that the model reads the data and set according to it 
import numpy as np
from sklearn.preprocessing import LabelEncoder
data=np.array(['male','female','male','female','male','female'])
le=LabelEncoder()
readed_data=le.fit_transform(data)
print(readed_data)
#System cannot khow the text from so change according to it 
