import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,confusion_matrix,roc_auc_score,roc_curve

# Create a simple dataset
X = np.array([
    [1,2],[2,3],[3,3],[4,5],[5,6],   # Class 0
    [6,8],[7,8],[8,9],[9,10],[10,11] # Class 1
])
y = np.array([0,0,0,0,0, 1,1,1,1,1])

mod=SVC(kernel='linear',probability=True)# for use of probability in SCV we should anable them
mod=mod.fit(X,y)
y_pred=mod.predict(X)
probability=mod.predict_proba(X)
print('Accuracy score ',accuracy_score(y,y_pred))
print('confusion matrix',confusion_matrix(y,y_pred))
print('Auc',roc_auc_score(y,probability[:,1]))
tpr,fpr,threshold=roc_curve(y,probability[:,1])
plt.plot(fpr,tpr,color='red',ls=':',label='Roc curve')
plt.plot([0,1],[0,1],color='black',ls='--',label='Mid line')
plt.xlabel("False rate ")
plt.ylabel("True rate")
plt.legend()
plt.show()
#-------------------cheaking the decision boundary--------------
plt.scatter(X[:,0], X[:,1], c=y, cmap='bwr', edgecolors='k')

w = mod.coef_[0]
b = mod.intercept_[0]
x_line = np.linspace(0,12,100)
y_line = -(w[0]/w[1])*x_line - b/w[1]
plt.plot(x_line, y_line, 'g--', label="Decision Boundary")
plt.legend()
plt.show()


