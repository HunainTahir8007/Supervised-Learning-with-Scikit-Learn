from sklearn.preprocessing import Binarizer

data = [[1.5], [2.5], [0.5]]
binarizer = Binarizer(threshold=1.0)
print(binarizer.fit_transform(data))

