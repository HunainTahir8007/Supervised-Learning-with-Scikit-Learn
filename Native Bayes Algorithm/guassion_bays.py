import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.naive_bayes import GaussianNB 
from sklearn.metrics import f1_score,accuracy_score,roc_auc_score,classification_report,confusion_matrix,roc_curve

pd.options.display.max_columns=None 
df=pd.read_csv("d:\\cssv\\bays.csv")

x=df.drop("Label",axis=1)
X=x.values
sc=StandardScaler()
X=sc.fit_transform(X)
y=df['Label'].values

X_train,X_test,Y_train,Y_test=train_test_split(X,y,train_size=0.8,random_state=42)
model=GaussianNB()
model.fit(X_train,Y_train)
y_pred=model.predict(X_test)
pro=model.predict_proba(X_test)
print("Accuracy score ",accuracy_score(Y_test,y_pred))
print("Confussion Matrix",confusion_matrix(Y_test,y_pred))
print('Classification report',classification_report(Y_test,y_pred))
print('f1 score',f1_score(Y_test,y_pred))
print("AUC",roc_auc_score(Y_test,y_pred))
#---------------------plotting the RoC curve--------------
fpr,tpr,threshold=roc_curve(Y_test,pro[:,1])
plt.plot(fpr,tpr,label='ROC Curve',color='red',ls=':')
plt.plot([0,1],[0,1],label='Middle',color='black',ls='--')
plt.legend(loc='best')
plt.grid()
plt.show()
#----------------------plotting the desion boundary---------
fe=X_train[:,2].mean()
xx,yy=np.meshgrid(
    np.linspace(X_train[:,0].min()-1,X_train[:,0].max()+1,300),
    np.linspace(X_train[:,1].min()-1,X_train[:,1].max()+1,300)
)
grid=np.c_[xx.ravel(),yy.ravel(),np.full(xx.ravel().shape,fe)]
mod=model.predict(grid)
mod=mod.reshape(xx.shape)
plt.contourf(xx, yy, mod, alpha=0.3, cmap=plt.cm.coolwarm)
plt.scatter(X_train[:,0], X_train[:,1], c=Y_train, cmap=plt.cm.coolwarm, s=30, edgecolors='k')
plt.title('Gusssian Native bayes')
plt.show()
