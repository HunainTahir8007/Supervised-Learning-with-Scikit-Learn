import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,roc_auc_score,roc_curve
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

data = {
    "feature1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "feature2": [2, 4, 1, 3, 7, 8, 6, 5, 9, 10],
    "feature3": [5, 3, 6, 2, 7, 9, 8, 4, 10, 1],
    "target":   [0, 0, 1, 0, 1, 1, 1, 0, 1, 0]
}
df=pd.DataFrame(data)
X=df[['feature1','feature2','feature3']].values
Y=df['target'].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
X_sca=StandardScaler()
X_train_scaled=X_sca.fit_transform(X_train)
X_tst_scaled=X_sca.transform(X_test)


model=SVC(kernel='poly',degree=3,C=100,gamma='scale',probability=True)
model.fit(X_train_scaled,Y_train)
y_pred=model.predict(X_tst_scaled)
pro=model.predict_proba(X_tst_scaled)
print("Accuracy score",accuracy_score(Y_test,y_pred))
print("Confusion matrix",confusion_matrix(Y_test,y_pred))
print('Classification report',classification_report(Y_test,y_pred))
print("AUC",roc_auc_score(Y_test,pro[:,1]))

#------------------testing---------------------------
my = np.array([[50,50,1]])   # 1 sample, 3 features
print("Shape of input:", my.shape)

final = X_sca.transform(my)  # keep shape (1, 3)
res = model.predict(final)

print("Prediction for my sample:", res)

#--------------------plotting the roc curve------------------
fpr,tpr,threshold=roc_curve(Y_test,pro[:,1])
plt.plot(fpr,tpr,label='ROC Curve',color='red',ls=':')
plt.plot([0,1],[0,1],label='Sepration line',color='black',ls='--')
plt.legend()
plt.show()
#------------------plottig model prediction -------------------
# Fix feature3 at its mean
feature3_mean = X_train_scaled[:,2].mean()

xx,yy=np.meshgrid(
    np.linspace(X_train_scaled[:,0].min()-1,X_train_scaled[:,0].max()+1,100),
    np.linspace(X_train_scaled[:,1].min()-1,X_train_scaled[:,1].max()+1,100),
)

grid = np.c_[xx.ravel(), yy.ravel(), np.full(xx.ravel().shape, feature3_mean)]
mod=model.predict(grid)
mod=mod.reshape(xx.shape)
plt.contourf(xx, yy, mod, alpha=0.3, cmap=plt.cm.coolwarm)
plt.scatter(X_train_scaled[:,0], X_train_scaled[:,1], c=Y_train, cmap=plt.cm.coolwarm, s=30, edgecolors='k')
plt.title("SVC with Poly Kernel - Decision Boundary (Feature3 fixed at mean)")
plt.show()