import seaborn as sb
import matplotlib.pyplot as plt
import matplotlib.animation as animation
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
print(len(x))
with open('data/q1/linearY.csv') as fp: 
    y=[]
    for line in fp:
        y.append(float(line[:-1]))      
print(len(y))



# In[96]:


X=np.array(x)
Y=np.array(y)
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

thetas= [0,0]


def grad_cost(X,theta, Y):
    temp = Y-np.dot(X,theta)
    temp2= temp * -X
    theta = (temp2.sum(axis=0))/(len(Y))
    return theta.T.reshape((theta.T.shape[0], 1))
    
def linear_reg(alpha):
    theta = np.zeros((2, 1))
    time_start = round(time.time(),3)
    for i in range(100000):
        gcost=grad_cost(X_norm, theta, Y)
        theta = theta - alpha * gcost
        if i%100==0 :
            thetas.append(theta[0][0])
            thetas.append(theta[1][0])
            time_start = round(time.time(),1)
    return theta


res = linear_reg(0.001)  
print(res)
print(cost(X_norm,res,Y))

thetas=np.array(thetas)
thetas = thetas.reshape(int(thetas.shape[0]/2), 2)
print(thetas.shape)


sb.set()
fig, ax = plt.subplots()
ax.plot(a,Y,'b^')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.title('x-y')
Y_dash =np.dot(X_norm,res)
print(Y_dash.shape)
plt.plot(a,Y_dash,color='red')
#for surfaceplot
theta0 = np.linspace(-0,2, 100)
theta1 = np.linspace(-1,1,100)
Theta0, Theta1 = np.meshgrid(theta0, theta1)
zs = np.array([cost(X_norm,np.array([[i],[j]]),Y) for i,j in zip(np.ravel(Theta0), np.ravel(Theta1))])
Cost = zs.reshape(Theta0.shape)

#for dynamic plot

actual_cost= np.array([cost(X_norm, np.reshape(thetas[i],(2,1)), Y) for i in range(thetas.shape[0])])
print(actual_cost.shape)

print('Done')


# In[133]:


fig = plt.figure(1)
ax = plt.axes(projection='3d')
ax.plot_surface(Theta0, Theta1, Cost, cmap=cm.coolwarm,linewidth=0,antialiased=False, alpha=0.6)
#Blues
#summer
# YlGnBu
#coolwarm

line, = ax.plot([],[],[],lw=2)

th0=[thetas[i][0] for i in range(thetas.shape[0])]
th1=[thetas[i][1] for i in range(thetas.shape[0])]
print(th0[:10])
print(th1[:10])


def animate(i):
    line.set_data(th0[:i],th1[:i])
    line.set_3d_properties(actual_cost[:i])
    return line,

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
anim = animation.FuncAnimation(fig, animate,frames=200, interval=200,
                           repeat_delay=5, blit=True)
# plt.show()



fig2 = plt.figure(2)
ax1=plt.axes()
theta0 = np.linspace(0.5,1.5, 100)
theta1 = np.linspace(-0.5,0.5,100)
Theta0, Theta1 = np.meshgrid(theta0, theta1)
zs = np.array([cost(X_norm,np.array([[i],[j]]),Y) for i,j in zip(np.ravel(Theta0), np.ravel(Theta1))])
Cost = zs.reshape(Theta0.shape)
ax1.contour(Theta0, Theta1, Cost)

line1, = ax1.plot([],[])

def animate2(i):
    line1.set_data(th0[:i],th1[:i])
    return line1,


anim2 = animation.FuncAnimation(fig2, animate2,frames=200, interval=200,repeat_delay=5, blit=True)

plt.show()
