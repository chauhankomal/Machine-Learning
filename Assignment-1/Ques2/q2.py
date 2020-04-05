import seaborn as sb
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import threading
import time
from matplotlib import cm


#(a) Sampling
with open('data/q2/q2test.csv') as fp: 
    x_test=[]
    y_test=[]
    for line in fp:
        line=line[:-1]
        [x_t1,x_t2,y_t]=line.split(',')
        x_test.append(float(x_t1))
        x_test.append(float(x_t2))
        y_test.append(float(y_t))
X_test1 = np.array(x_test).reshape(int(len(x_test)/2),2)
X_test = np.append(np.ones((X_test1.shape[0], 1)), X_test1, axis=1)
Y_test= np.array(y_test).reshape(np.array(y_test).shape[0],1)

x1= np.random.normal(3, 2, 1000000)
x2= np.random.normal(-1, 2, 1000000)
e= np.random.normal(0, np.sqrt(2), 1000000)
Y1= 3+ x1+ 2*x2+ e
Y = Y1.reshape(Y1.shape[0],1)

X_temp = np.append(np.ones((1000000,1)), x1.reshape(1000000,1) ,axis=1)
X_temp2 =np.append(X_temp, x2.reshape(1000000,1), axis=1)

X_Y= np.append(X_temp2, Y, axis=1)
np.random.shuffle(X_Y)
X= X_Y[:,0:3]
Y=X_Y[:,3:4]


#(b) Implementation of Stochastic Gradent Descent

def cost(X1, theta1, Y1):
    return (np.sum((np.dot(X1, theta1) - Y1) ** 2) /(2* len(Y1)))


def grad_cost(X,theta, Y):
    temp = Y-np.dot(X,theta)
    temp2= temp * -X
    theta = (temp2.sum(axis=0))/(len(Y))
    return theta.T.reshape((theta.T.shape[0], 1))


def sto_gradient(X,Y,thetas,r, diff, steps, avg_no):
    total_itr=flag =0
    avg_th=[]
    theta = np.zeros((3, 1))
    prev_cost = cost(X[:r,:], theta, Y[:r,:])
    for  j in range(steps):
        for i in range(0,1000000,r):
            total_itr =total_itr+1
            X_batch = X[i:r+i,:] 
            Y_batch= Y[i:r+i,:]
            gcost=grad_cost(X_batch, theta, Y_batch)
            theta = theta - 0.001 * gcost
            if i%(r*10)==0 :
                thetas.append(theta[0][0])
                thetas.append(theta[1][0])
                thetas.append(theta[2][0])
            curr_cost=  cost(X_batch, theta, Y_batch)
            avg_th.append(curr_cost)
            if(total_itr % avg_no == 0):                
                cost_diff = prev_cost-np.mean(avg_th)
                prev_cost = np.mean(avg_th)     
                avg_th =[]
                if abs(cost_diff) <= diff and total_itr > 1000000/r:
                    flag=1
                    break
        if flag==1:
            break
    return theta ,total_itr, thetas

def animate(i, th0, th1, th2,line):
    line.set_data(th0[:i],th1[:i])
    line.set_3d_properties(th2[:i])
    return line,

def plot(thetas1): 
    fig = plt.figure(figsize=(8, 6), dpi= 80)
    ax = plt.axes(projection='3d')
    thetas=np.array(thetas1)
    thetas = thetas.reshape(int(thetas.shape[0]/3), 3)
    th0=[thetas[i][0] for i in range(thetas.shape[0])]
    th1=[thetas[i][1] for i in range(thetas.shape[0])]
    th2=[thetas[i][2] for i in range(thetas.shape[0])]
    ax.set_xlim3d(0, 3.2)
    ax.set_ylim3d(0,1.5)
    ax.set_zlim3d(0,2)
    line, = ax.plot([],[],[],lw=2)
    ax.set_xlabel('theta0')
    ax.set_ylabel('theta1')
    ax.set_zlabel('theta3')
    anim = animation.FuncAnimation(fig, animate,frames=len(th0), fargs=(th0, th1, th2, line), interval=40,
                           repeat_delay=1, blit=True)
    plt.show();
    return anim,ax


# Batch size r =1
# **parameters learned**
# $\theta_0 = 2.9480744 $
# $\theta_1 = 1.01232689$
# $\theta_2 = 2.01740026$
 
# **Diffrence in parameters**
# $\theta_0 = 0.0519256 $
# $\theta_1 = 0.01232688$
# $\theta_2 = 0.0174$

# **Time Taken** 69.81 seconds
# **N0. of iterations**- 1064000
# **Cost -** 1.002598692

# **Convergence Criteria**
# First traverse the whole m examples atleast once then
# taken average cost after every 1000 iterations an d compare it with prev average cost\
# if $J(\theta^{(i)}) - J(\theta^{(i+1)}) <= 0.0001$ then converged
# Also set upper bound on number of iterations of data so if it doesn't pass convergence criteria then it will stop after 1000    

theta= np.ones((3,1))
thetas= [0,0,0]
start_time =time.time()
res1, itr, thetas = sto_gradient(X,Y,thetas,1,0.0001,1000,1000)
print('Time',time.time()-start_time)
print('Iterations- ',itr)
print(res1)   
print('cost ',cost(X,res1, Y))
print('Done')

plot(thetas)


# ## Batch size r =100

# **parameters learned**
# $\theta_0 = 2.99737108 $
# $\theta_1 = 0.99731907$
# $\theta_2 = 2.00097804$

# **Difference in parameters**
# $\theta_0 =  0.00262891$
# $\theta_1 = 0.00268$
# $\theta_2 = 0.00097804 $

# **Time Taken** 86.4108 seconds
# **N0. of iterations**- 1000000
# **Cost - 1.0013732546240643**

# **Convergence criteia**\
# First traverse the whole m examples atleast once then
# taken average cost after every 1000 iterations an d compare it with prev average cost\
# if $J(\theta^{(i)}) - J(\theta^{(i+1)}) <= 0.0001$ then converged

theta= np.ones((3,1))
thetas= [0,0,0]
start_time =time.time()
res2, itr, thetas = sto_gradient(X,Y,thetas,100,0.0001,100, 1000)
print('start')
print('Time',time.time()-start_time)
print('Iterations- ',itr)
print(res2)   
print('cost ',cost(X,res2, Y))
print('Done')

plot(thetas)


# Batch size r =10000

# **parameters learned**
# $\theta_0 = 2.96347761$
# $\theta_1 = 1.01232689$
# $\theta_2 = 0.01740026$

# **Difference in parameters**
# $\theta_0 =  0.03652239$
# $\theta_1 = 0.01232688$
# $\theta_2 = 0.00097804 $

# **Time Taken** 19.9656553 seconds
# **N0. of iterations**- 1064000
# **Error -** 1.002598692

# **Convergence Criteria**
# First traverse the whole m examples atleast once then
# taken average cost after every 1000 iterations an d compare it with prev average cost\
# if $J(\theta^{(i)}) - J(\theta^{(i+1)}) <= 0.00001$ then converged

theta= np.ones((3,1))
thetas= [0,0,0]
start_time =time.time()
res3, itr, thetas = sto_gradient(X,Y,thetas,10000,0.00001,1000, 100)
print('start')
print('Time',time.time()-start_time)
print('Iterations- ',itr)
print(res3)   
print('cost ',cost(X,res3, Y))
print('Done')

plot(thetas)

# Batch size r =1000000

# **parameters learned**
# $\theta_0 = 2.890994271$\
# $\theta_1 = 1.02368121$\
# $\theta_2 = 1.99112236$

# **Time Taken** 19.9656553 seconds
# **N0. of iterations**- 11770
# **Error -** 1.00052844954
 
# **Convergence Criteria**
# First traverse the whole m examples atleast once then
# taken average cost after every 1000 iterations an d compare it with prev average cost\
# if $J(\theta^{(i)}) - J(\theta^{(i+1)}) <= 10^{-6}$ then converged

theta= np.ones((3,1))
thetas= [0,0,0]
start_time =time.time()
print('start')
res4, itr , thetas= sto_gradient(X,Y,thetas,1000000,0.000001,50000, 1)
print('Time',time.time()-start_time)
print('Iterations- ',itr)
print(res4)   
print('cost ',cost(X,res4, Y))
print('Done')

plot(thetas)


# (c) Observation
# 1 iteration = linear regression on 1 batch \
# when batch size is small it takes more iteration to converge as compared to large batch size \
# $No.\;of\;iterations \propto \frac{1}{batch\;size}$
# 
# It converges fast when batch size is 10000 and takes too long to converge when batch size is 1000000.

# Error on test data
# Cost when batch size = 1
# **$1.004445895587165$**

cost1 = cost(X_test, res1, Y_test)


# ### Cost when batch size = 100
# **$0.9834276129619844$**

cost2 = cost(X_test, res2, Y_test)


# ### Cost when batch size = 10000
# **$0.9863700639920684$**

cost3 = cost(X_test, res3, Y_test)


# ### Cost when batch size = 1000000
# **$1.0180463660973345$**

cost4 = cost(X_test, res4, Y_test)


# (e) Observation on plot of theta
# When batch size is too small (r = 1) then plot is noisy as it update parameters based on only one example,
# as we increase batch size then plot becomes more smoother because parameters are updated based on some subset of data which gives enough information about data.

