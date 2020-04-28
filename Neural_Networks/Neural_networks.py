import numpy as np 
import pandas as pd
import math
import sys
from time import time
from matplotlib import pyplot as plt
from sklearn.neural_network import MLPClassifier
from NNClassifier import NNClassifier

def process_data(data):
    np.random.shuffle(data)
    X= data[:, :-1]/255
    y = data[:, -1]
    n_classes= len(set(y))
    Y = np.zeros((y.shape[0], n_classes))
    for i in range(y.shape[0]):
        Y[i][y[i]]=1
    return X, Y

def plot_acc(x, y1,y2,t):
    ax= plt.figure(figsize=(12,7))
    plt.plot(x,y1)
    plt.plot(x,y2)
    plt.xlabel('Number of units in hidden layer')
    plt.ylabel("Accuracy")
    plt.title(t)
    plt.legend(['train accuracy', 'test accuracy'])
    plt.show()

def plot_time(x, y,t):
    ax= plt.figure(figsize=(12,7))
    plt.plot(x,y)
    plt.title(t)
    plt.xlabel('Number of units in hidden layer')
    plt.ylabel('Time taken (in min.)')
    plt.show()


train_data = np.array(pd.read_csv(sys.argv[1], header=None))
test_data = np.array(pd.read_csv(sys.argv[2], header=None))
train_X, train_Y = process_data(train_data)
test_X , test_Y = process_data(test_data)

layers =[1,5,10,50,100]
test_accuracies =[]
train_accuracies =[]
time_taken =[]
for i in layers:
      model = NNClassifier(batch_size=100, n_features=784, hidden_layers=[i], n_classes=26, activation_fun='sigmoid', max_iter=2000, stop = 1e-8)
      s = time()
      model.fit(train_X, train_Y)
      print('time taken ',(time()-s)/60)
      time_taken.append((time()-s)/60)
      train = model.score(train_X, train_Y)*100
      test = model.score(test_X, test_Y)*100
      train_accuracies.append(train)
      test_accuracies.append(test)
      print('train accuracy when hidden units = ',i,train)
      print('test accuracy when hidden units = ',i,test)

plot_time(layers, time_taken,'Time taken in training NN with constant learning rate')
plot_acc(layers,train_accuracies,test_accuracies,'Accuracies with constant learning rate')

"""## Adaptive learning"""

test_accuracies1 =[]
train_accuracies1 =[]
time_taken1 =[]
layers =[1,5,10,50,100]
for i in layers:
    model = NNClassifier(batch_size=100, n_features=784, hidden_layers=[i], n_classes=26, activation_fun='sigmoid', alpha=0.5, learning='adaptive', max_iter =3000, stop =1e-8)
    s = time()
    model.fit(train_X, train_Y)
    print('time taken ',(time()-s)/60)
    time_taken1.append((time()-s)/60)
    train = model.score(train_X, train_Y)*100
    test = model.score(test_X, test_Y) *100
    test_accuracies1.append(test)
    train_accuracies1.append(train)
    print('train accuracy when hidden units = ', i,', ',train)
    print('test accuracy when hidden units = ', i,', ',test)


plot_time(layers, time_taken1,'Time taken in training NN with adaptive learning rate')
plot_acc(layers,train_accuracies1,test_accuracies1,'Accuracies with constant learning rate' )

"""# (d)RELU"""


model = NNClassifier(batch_size=100, n_features=784, hidden_layers=[100,100], n_classes = 26, activation_fun='relu', alpha=0.5, learning='adaptive', max_iter =3000, stop =1e-8)
s = time()
model.fit(train_X, train_Y)
print('time taken ',(time()-s)/60)
train_acc = model.score(train_X, train_Y)
test_acc = model.score(test_X, test_Y)  
print('train accuracy with ReLU = ',train_acc)
print('test accuracy with ReLU = ',test_acc)


model = NNClassifier(batch_size=100, n_features=784, hidden_layers=[100,100], n_classes = 26, activation_fun='sigmoid', alpha=0.5, learning='adaptive', max_iter =3000, stop =1e-8)
s = time()
model.fit(train_X, train_Y)
print('time taken ',(time()-s)/60)
train_acc = model.score(train_X, train_Y)
test_acc = model.score(test_X, test_Y)  
print('train accuracy with Sigmoid = ',train_acc)
print('test accuracy with Sigmoid = ',test_acc)


"""## (e) scikit Learn"""


s = time()
MLP = MLPClassifier(hidden_layer_sizes=[100,100], batch_size = 100, solver ='sgd',learning_rate_init=0.5,activation='relu',learning_rate ='invscaling', max_iter=3000, momentum =0,tol=1e-8)
MLP.fit(train_X, train_Y)
print((time()-s)/60)

print('Train Accuracy with ReLU',MLP.score(train_X, train_Y))
print('Test Accuracy with ReLU',MLP.score(test_X,test_Y))
print('Number of epochs ',MLP.n_iter_)

s = time()
MLP = MLPClassifier(hidden_layer_sizes=[100,100],batch_size=100, learning_rate_init = 0.5,solver ='sgd',activation='logistic',learning_rate ='invscaling', max_iter = 3000)
MLP.fit(train_data[:,:-1]/255, train_data[:,-1])
print((time()-s)/60)

print('Train Accuracy with Sigmoid',MLP.score(train_data[:,:-1]/255, train_data[:,-1]))
print('Test Accuracy with Sigmoid',MLP.score(test_data[:,:-1]/255,test_data[:,-1]))
print('Number of epochs ',MLP.n_iter_)