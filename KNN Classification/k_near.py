import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,roc_auc_score,roc_curve
from sklearn.neighbors import KNeighborsClassifier

pd.options.display.max_columns=None
df=pd.read_csv('d:\\cssv\\knn.csv')
X=df.drop(columns=['Unnamed: 32','id','diagnosis']).values
Y=df['diagnosis'].values
print(df)

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=0.8,random_state=42)

X_sca=StandardScaler()
X_train_scaled=X_sca.fit_transform(X_train)
X_test_scaled=X_sca.transform(X_test)

Y_encoder=LabelEncoder()
Y_train_encoded=Y_encoder.fit_transform(Y_train)
Y_test_encoded=Y_encoder.transform(Y_test)
#------------------------------cheaking best K value-------------------------
acc=[]
for i in range(1,20):
    model=KNeighborsClassifier(n_neighbors=i)
    model.fit(X_train_scaled,Y_train_encoded)
    Y_pred=model.predict(X_test_scaled)
    Accuracy=accuracy_score(Y_test_encoded,Y_pred)
    acc.append(Accuracy)
#-----------------------------------------------------------------------------------

model=KNeighborsClassifier(n_neighbors=9)
model.fit(X_train_scaled,Y_train_encoded)
Y_pred=model.predict(X_test_scaled)
pro=model.predict_proba(X_test_scaled)
Accuracy=accuracy_score(Y_test_encoded,Y_pred)
print("Accuracy score ",Accuracy)
con=confusion_matrix(Y_test_encoded,Y_pred)
print("Confusion matrix",con)
clas=classification_report(Y_test_encoded,Y_pred)
print("Classificaation Report",clas)
auc=roc_auc_score(Y_test_encoded,pro[:,1])
print("AUC",auc)
#-----------------------testing the Model---------------------------------------
my=np.array([[7.76,24.54,47.92,181,0.05263,0.04362,0,0,0.1587,0.05884,0.3857,1.428,2.548,19.15,0.007189,0.00466,0,0,0.02676,0.002783,9.456,30.37,59.16,268.6,0.08996,0.06444,0,0,0.2871,0.07039]])
my_tr=X_sca.transform(my)
prediction=model.predict(my_tr)
final=Y_encoder.inverse_transform(prediction)
print('Tumor is ',final)
#-----------------------------plotting the ROC Curve-----------------------------
fpr,tpr,threshold=roc_curve(Y_test_encoded,pro[:,1])
plt.plot(fpr,tpr,label='Roc Curve',color='red',ls=":")
plt.plot([0,1],[0,1],color='black',label='Sepreation line',ls='--')
plt.title("ROC Curve Movement")
plt.legend(loc='best')
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.show()

#----------------------------best boundary line---------------------------------

f1, f2 = 0, 1   

xx, yy = np.meshgrid(
    np.linspace(X_train_scaled[:,f1].min()-1, X_train_scaled[:,f1].max()+1, 100),
    np.linspace(X_train_scaled[:,f2].min()-1, X_train_scaled[:,f2].max()+1, 100)
)


X_mean = X_train_scaled.mean(axis=0)
grid = np.tile(X_mean, (xx.ravel().shape[0], 1))

grid[:, f1] = xx.ravel()
grid[:, f2] = yy.ravel()

Z = model.predict(grid)
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
plt.scatter(X_train_scaled[:,f1], X_train_scaled[:,f2], c=Y_train_encoded, edgecolors='k')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Decision boundary (with other features fixed at mean)")
plt.show()