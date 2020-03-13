
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

import cvxopt
import csv
import pandas as pd
import numpy as np
from random import randint
import time
from tqdm import tqdm, tqdm_notebook
import math
import pickle
from cvxopt import solvers
from cvxopt import matrix
from scipy.spatial.distance import pdist, cdist
from sklearn.svm import SVC

train= pd.read_csv('fashion_mnist/train.csv',header=None).to_numpy()
val= pd.read_csv('fashion_mnist/val.csv',header=None).to_numpy()
test= pd.read_csv('fashion_mnist/test.csv',header=None).to_numpy()

def load_data(a,b):
    train_data = np.array([train[i,:] for i in range(train.shape[0]) if train[i,-1]== a or train[i,-1]==b ] )
    val_data = np.array([val[i,:] for i in range(val.shape[0]) if val[i,-1]==a or val[i,-1]==b ] )
    test_data = np.array([test[i,:] for i in range(test.shape[0]) if test[i,-1]==a or test[i,-1]==b ] )
    
    X_val = val_data[:, :-1]/255
    m_val= X_val.shape[0]
    Y1_val = val_data[:,-1].reshape((m_val,1))
    Y1_val[Y1_val==a] = -1
    Y1_val[Y1_val==b] = 1
    
    X_test = test_data[:, :-1]/255
    m_test= X_test.shape[0]
    Y1_test = test_data[:,-1].reshape((m_test,1))
    Y1_test[Y1_test==a] = -1
    Y1_test[Y1_test==b] = 1
    
    X = train_data[:, :-1]/255
    m= X.shape[0]
    Y1 = train_data[:,-1].reshape((m,1))
    Y1[Y1==a] = -1
    Y1[Y1==b] = 1
    
    return X,Y1,X_val, Y1_val, X_test, Y1_test

def svm(X,Y,c):
    m= X.shape[0]
    XX = np.dot(X, X.T)
    YY = Y@Y.T
    P=matrix(np.multiply(XX,YY))
    q=matrix(np.ones([m,1])*-1)
    I = np.identity(m)
    I_neg = np.identity(m)*-1
    G = matrix(np.concatenate([I,I_neg]))
    A=matrix(Y.reshape((1,m)))
    b = matrix([0], tc='d')
    Y=Y.reshape(m)
    h = matrix(np.concatenate((np.ones([m,1])*C,np.zeros([m,1]))))  
    solvers.options['show_progress'] = False
    sol = solvers.qp(P,q,G,h,A,b)
    alpha=np.array(sol['x'])
    SV_i =[i for i in range(len(alpha)) if alpha[i] >=1e-5]
    alpha1= np.array([alpha[i] for i in SV_i])
    X1 = np.array([X[i] for i in SV_i])    
    Y1 = np.array([Y[i] for i in SV_i]).reshape((X1.shape[0],1))
    w =np.dot(X1.T, (alpha1*Y1))
    X_pos= [X[i] for i in range(len(Y)) if Y[i]==1]
    X_neg= [X[i] for i in range(len(Y)) if Y[i]==-1]
    temp1 = np.dot(X_pos, w)
    temp2 = np.dot(X_neg, w)
    b = -(temp1.min()+temp2.max())/2
    return w,b,len(SV_i)


def find_accuracy(y, y_pred):
    correct =0
    
    for i in range(len(y)): 
        if y[i] == y_pred[i]:
            correct +=1
    return (correct/len(y)) *100

def predict(w, b, X):
    pred = np.dot(X,w)+b
    Y_pred =[]
    for i in pred :
        if i >=0:
            Y_pred.append(1)
        else :
            Y_pred.append(-1)
    return Y_pred

X,Y,X_val, Y_val, X_test, Y_test= load_data(4,5)
s = time.time()
C = 1.0
w, b, nSV = svm(X, Y,C)
print("No of support Vectors -", nSV)
print("b ", b)
print('Time taken in svm using linear kernel ', time.time()-s)

Y_pred= predict(w,b,X)
accuracy = find_accuracy(Y, Y_pred)
print('Accuracy on Training data -',accuracy)

Y_pred_val = predict(w,b,X_val)
accuracy = find_accuracy(Y_val, Y_pred_val)
print('Accuracy on validation set  -',accuracy)

Y_pred_test = predict(w,b,X_test)
accuracy = find_accuracy(Y_test, Y_pred_test)
print('Accuracy on test data  -',accuracy)


# ## 1(b) SVM Using Gaussian Kernel

def svm_gaussian(X,Y,C, gamma):
    m= X.shape[0]
    t =cdist(X,X,'euclidean')
    K = np.exp((t**2) * -gamma)
    YY = Y@Y.T
    P=matrix(np.multiply(K,YY))
    q=matrix(np.ones([m,1])*-1)
    I = np.identity(m)
    I_neg = np.identity(m)*-1
    G = matrix(np.concatenate([I,I_neg]))
    A=matrix(Y.reshape((1,m)))
    b = matrix([0], tc='d')
    Y=Y.reshape(m)
    h = matrix(np.concatenate((np.ones([m,1])*C,np.zeros([m,1]))))  
    sol = solvers.qp(P,q,G,h,A,b)
    alpha=np.array(sol['x'])   
    SV_i =[i for i in range(len(alpha)) if alpha[i] >=1e-4]
    alpha1= np.array([alpha[i] for i in SV_i])
    SV_X = np.array([X[i] for i in SV_i])
    SV_Y = np.array([Y[i] for i in SV_i]).reshape((SV_X.shape[0],1))   
    X_pos= np.array([X[i] for i in range(len(Y)) if Y[i]==1])
    X_neg= np.array([X[i] for i in range(len(Y)) if Y[i]==-1])
    t_neg= cdist(SV_X,X_neg,'euclidean')
    K_neg= np.exp((t_neg**2) * (-0.05))
    t_pos= cdist(SV_X,X_pos,'euclidean')
    K_pos= np.exp((t_pos**2) * (-0.05))
    X_neg_k =np.dot(K_neg.T,(SV_Y*alpha1))
    X_pos_k =np.dot(K_pos.T,(SV_Y*alpha1))
    b = -(X_neg_k.max()+X_pos_k.min())/2
    return alpha1,b,SV_X, SV_Y

def predict_gaussian(alpha,b,SV_X, SV_Y,X):
    t= cdist(SV_X,X,'euclidean')
    K= np.exp((t**2) * (-0.05))
    pred = np.dot(K.T,(SV_Y*alpha))+b
    Y_pred =[]
    for i in pred :
        if i >0:
            Y_pred.append(1)
        else :
            Y_pred.append(-1)
    return Y_pred

s = time.time()
C=1.0
gamma = 0.05
alpha,b,SV_X, SV_Y = svm_gaussian(X,Y,C,gamma)
print('Number of support vectors -',SV_X.shape[0])
print('b', b)
print('time taken in svm gaussian kernel training-', time.time()-s)

Y_pred= predict_gaussian(alpha,b,SV_X, SV_Y,X)
acc = find_accuracy(Y, Y_pred)
print('Accuracy on training set using gaussian kernel -', acc )

Y_pred_val =predict_gaussian(alpha,b,SV_X, SV_Y,X_val)
accuracy = find_accuracy(Y_val, Y_pred_val)
print('Accuracy on validation set using gaussian kernel -', accuracy )

Y_pred_test = predict_gaussian(alpha,b,SV_X, SV_Y,X_test)
acc = find_accuracy(Y_test, Y_pred_test)
print('Accuracy on test set using gaussian kernel -', acc )


# # 1(c) Scikit SVM

# ## Linear kernel

s = time.time()
svc = SVC(kernel='linear')
svc.fit(X,np.ravel(Y ,order = 'C') )
print('NUmber of support vetors ', len(svc.support_vectors_) )
print('b',svc.intercept_)
print('time taken in scikit learn svm linear kernel training-', time.time()-s)

y_pred = svc.predict(X)
acc=find_accuracy(Y,Y_pred)
print('Accuracy of training set using sklearn linear kernel-', acc)

Y_val_pred = svc.predict(X_val)
acc = find_accuracy(Y_val,Y_val_pred)
print('Accuracy of validation set using sklearn linear kernel -', acc)

Y_test_pred = svc.predict(X_test)
acc=find_accuracy(Y_test,Y_test_pred)
print('Accuracy of test set using sklearn linear kernel-', acc)


# ## Gaussian Kernel

s = time.time()
svc = SVC(kernel='rbf', gamma=0.05, C=1)
t = svc.fit(X, np.ravel(Y ,order = 'C'))
print('NUmber of support vetors ', len(svc.support_vectors_) )
print('b',svc.intercept_)
print('Time taken in scikit learn svm gaussian kernel training-', time.time()-s)

y_pred = svc.predict(X)
acc=find_accuracy(Y,Y_pred)
print('Accuracy of training set using sklearn(gaussian kernel)-', acc)

Y_val_pred = svc.predict(X_val)
acc = find_accuracy(Y_val,Y_val_pred)
print('Accuracy of validation set using sklearn(gaussian kernel) -', acc)

Y_test_pred = svc.predict(X_test)
acc=find_accuracy(Y_test,Y_test_pred)
print('Accuracy of test set using sklearn(gaussian kernel) -', acc)


