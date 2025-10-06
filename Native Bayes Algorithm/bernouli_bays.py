import numpy as np 
import pandas  as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import BernoulliNB
import sys
import io
from sklearn.metrics import accuracy_score,roc_auc_score,roc_curve,f1_score,confusion_matrix,classification_report,auc
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.decomposition import TruncatedSVD

pd.options.display.max_columns=None
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
df=pd.read_csv("d:\\cssv\\spam_or_not_spam.csv")
df.dropna(inplace=True)
df.reset_index(inplace=True)
x=df['email']
X=x.values
vc=CountVectorizer()
X=vc.fit_transform(X)
y=df['label']
Y=y.values

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=0.8,random_state=42)
mod=BernoulliNB(alpha=0.01,binarize=0.0)
mod.fit(X_train,Y_train,)
y_pred_test=mod.predict(X_test)
y_pred_train=mod.predict(X_train)
print("Acc test",accuracy_score(Y_test,y_pred_test))
print('acc train',accuracy_score(Y_train,y_pred_train))
pro=mod.predict_proba(X_test)
print("F1 score",f1_score(Y_test,y_pred_test))
print("Classification report",classification_report(Y_test,y_pred_test))
print("Confusiion matrix ",confusion_matrix(Y_test,y_pred_test))
print("AUC score",roc_auc_score(Y_test,pro[:,1]))
#------------------cheaking the best hyperparameters--------------
bnb = BernoulliNB()
param_grid = {
    'alpha': [0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
    'binarize': [None, 0.0]   
}

grid = GridSearchCV(bnb, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, Y_train)


print("Best Parameters:", grid.best_params_)
print("Best CV Accuracy:", grid.best_score_)
#----------------------ROC curve-----------------------------------
fpr, tpr, _ = roc_curve(Y_test, pro[:,1])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, color="red", ls=":", label="ROC Curve (AUC = %0.2f)" % roc_auc)
plt.plot([0,1], [0,1], color="black", ls="--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - BernoulliNB")
plt.legend(loc="lower right")
plt.grid()
plt.show()
#---------------------------------------------------
#------------plotting the best boundary------------
#Note:-
     #we have only 1 column so we have to convert it into the 2d 
tr=TruncatedSVD(n_components=2,random_state=42)
tr_train=tr.fit_transform(X_train)


model=BernoulliNB(alpha=0.01,binarize=0.0)
model.fit(tr_train,Y_train)

xx,yy_mesh=np.meshgrid(
    np.linspace(tr_train[:,0].min()-1,tr_train[:,0].max()+1 ,300),
    np.linspace(tr_train[:,1].min()-1,tr_train[:,1].max()+1 ,300)
)
grid= np.column_stack((xx.ravel(),yy_mesh.ravel()))

pre=model.predict(grid)
pre=pre.reshape(xx.shape)
plt.contourf(xx, yy_mesh, pre, alpha=0.3, cmap=plt.cm.coolwarm)
plt.scatter(tr_train[:,0], tr_train[:,1], c=Y_train, edgecolor="k", s=30)
plt.xlabel("SVD Component 1")
plt.ylabel("SVD Component 2")
plt.title("BernoulliNB Decision Boundary (2D Reduced Features)")
plt.show()
#-------------------------------------------------------------