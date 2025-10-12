import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,roc_auc_score,roc_curve,classification_report,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.datasets import make_moons
dat=make_moons()
pd.options.display.max_columns=None
X, y = make_moons(n_samples=300, noise=0.2, random_state=42)


df = pd.DataFrame(X, columns=['feature1', 'feature2'])
df['target'] = y  

X=df[['feature1','feature2']].values
Y=df['target'].values

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=0.8,random_state=42)

X_sca=StandardScaler()
X_train_sca=X_sca.fit_transform(X_train)
X_test_sca=X_sca.transform(X_test)

model=SVC(kernel='rbf',gamma='scale',C=100,probability=True)
mod=model.fit(X_train_sca,Y_train)
Y_pred=mod.predict(X_test_sca)
pro=mod.predict_proba(X_test_sca)

acc=accuracy_score(Y_test,Y_pred)
print("Accuracy score",acc)
con=confusion_matrix(Y_test,Y_pred)
print("Confusion matrix",con)
cla=classification_report(Y_test,Y_pred)
print("Classification Report",cla)
auu=roc_auc_score(Y_test,pro[:,1])
print("AUC Score",auu)

#--------------------------ploting----------------------
fpr,tpr,threshold=roc_curve(Y_test,pro[:,1])
plt.plot(fpr,tpr,label='ROC Curve',color='red',ls=':')
plt.plot([0,1],[0,1],label='Sepration line',color='black',ls='--')
plt.legend()
plt.show()
#------------------------------------------------------------------------------------=-=-=
xx, yy = np.meshgrid(
    np.linspace(X_train_sca[:,0].min()-1, X_train_sca[:,0].max()+1, 500),
    np.linspace(X_train_sca[:,1].min()-1, X_train_sca[:,1].max()+1, 500)
)

# Predict class for each grid point
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot contour + training points
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
plt.scatter(X_train_sca[:,0], X_train_sca[:,1], c=Y_train, cmap=plt.cm.coolwarm, s=30, edgecolors='k')
plt.title("SVC with RBF Kernel - Decision Boundary")
plt.show()