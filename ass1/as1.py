import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import threading
import time
from matplotlib import cm


with open('data/q1/linearX.csv') as fp: 
    x=[]
    for line in fp:
        x.append(float(line[:-1]))
#print(X)      
print(len(x))
with open('data/q1/linearY.csv') as fp: 
    y=[]
    for line in fp:
        y.append(float(line[:-1]))
#print(Y)      
# print(len(y))
X=np.array(x)
Y=np.array(y)
# print(X.shape)
# print(Y.shape)
X = X.reshape((X.shape[0], 1))
Y = Y.reshape((Y.shape[0], 1))
X_mean = np.mean(X)
X_var= np.sum((X-X_mean)**2)/len(X)
X_std_dev= np.sqrt(X_var)
X_norm = (X- X_mean)/X_std_dev



X_norm = np.append(np.ones((X_norm.shape[0], 1)), X_norm, axis = 1)
a=X_norm[:,1:2]
theta = np.zeros((2, 1))

def cost(X, theta, Y):
    return np.sum((np.dot(X, theta) - Y) ** 2)/(2*len(Y))

# thetas= np.array()
# def store(theta):
#     thetas.append(theta)
        
def grad_cost(X,theta, Y):
    temp = Y-np.dot(X,theta)
    temp2= temp * -X
    theta = (temp2.sum(axis=0))/(len(Y))
    return theta.T.reshape((theta.T.shape[0], 1))
    
def linear_reg(alpha):
    theta = np.zeros((2, 1))
    time_start = round(time.time(),1)
    for i in range(100000):
#         if i < 10:
#             print(theta)
#             #print(cost(X_norm, theta, Y),'---',theta)
        theta = theta - alpha * grad_cost(X_norm, theta, Y)
    if(round(time.time()-time_start),1)==0.2 :
        store(theta)
    return theta



# print(X.shape)  
# print(thetas)
res = linear_reg(0.1)  
# print(res)
# print(cost(X_norm,res,Y))



# theta0 = np.linspace(-1,1, 1000)
# theta1 = np.linspace(-1,1,1000)
# cost_value = np.array([cost(X_norm,np.array([[i],[j]]),Y)  for i in theta0 for j in theta1])
# print(cost_value[500:510])
# print(cost_value.shape)



# sb.set()
fig, ax = plt.subplots()
ax.plot(a,Y,'b^')
# ax.scatter(a,Y)
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.title('x-y')
Y_dash =np.dot(X_norm,res)
print(Y_dash.shape)
plt.plot(a,Y_dash,color='red')
plt.show()

theta0 = np.linspace(-0.5,2.5,80)
theta1 = np.linspace(-1.5,1.5,80)
Theta0, Theta1 = np.meshgrid(theta0, theta1)
zs = np.array([cost(X_norm,np.array([[i],[j]]),Y) for i,j in zip(np.ravel(Theta0), np.ravel(Theta1))])
Cost = zs.reshape(Theta0.shape)
print('Done')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# theta0 = theta1 = np.arange(-3.0, 3.0, 0.05)

ax.plot_surface(Theta0, Theta1, Cost, cmap=cm.coolwarm,rstride=1, cstride=1, color='b', alpha=0.5,antialiased=False)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()
