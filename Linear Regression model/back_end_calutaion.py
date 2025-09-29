#This is the code for which actual mathematical operation in done behind the linear Regression
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

Week=[1,2,3,4,5]
Sales=[2,4,5,4,5]

df=pd.DataFrame({"week":Week,'Sales':Sales})
X=df['week'].values
Y=df['Sales'].values
print(df)
mean_x=np.mean(X)
mean_y=np.mean(Y)

#As we khow the Formula
# y=mx+c 
#Firstly we find the m
# M Formula :--
#  m=(x-x')(y-y')/(x-x')**2
n=len(X)
num=0
deno=0

for i in range(n):
    num=num+(X[i]-mean_x) *(Y[i] - mean_y)
    deno=deno+(X[i]-mean_x)**2

m=num/deno
# now finding the  c 
# y =mx+c   ///c=y'-mx
c  =mean_y - (m * mean_x)
print(f'M {m}')
print(f'C {c}')

max_x=np.max(X)+ 1
min_x=np.min(X) - 1
x=np.linspace(max_x,min_x) #this crete the 50 random numbers between the X and Y
y=(m*x )+ c
print(x)
print(y)

plt.plot(x,y,linestyle="--",color='red',label="-- for Regression") # pridcted value 
plt.scatter(X,Y,label='for the actual values ',color='black')# for the actual line 
plt.grid(linestyle=':',color='grey',linewidth=0.2)
plt.legend(loc='best')
plt.show()

#now finding the value of the r2

no=0
den=0

for i in range(n):
    yy=(m*X[i]) + c
    no=no+(Y[i]-yy)**2
    den=den+(Y[i]-mean_y)  **2

r2=1-(no/den)  
print(r2)
