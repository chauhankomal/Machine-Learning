import numpy as np 
import pandas as pd
import math

class NNClassifier:
    
    def __init__(self,batch_size, n_features , n_classes ,hidden_layers=[100], activation_fun = 'sigmoid', alpha=0.1,learning='constant', max_iter=2000, stop = 1e-8):
        self.hidden_layers= hidden_layers
        self.batch_size = batch_size
        self.hidden_layers= hidden_layers
        self.n_features= n_features
        self.n_classes= n_classes
        self.alpha = alpha
        self.learning = learning
        self.max_iter = max_iter
        self.stop = stop
        self.activation_fun= activation_fun
        self.n_layers = len(self.hidden_layers)+2


    def shuffle_data(self):
        temp = np.concatenate((self.X, self.Y), axis=1)
        np.random.shuffle(temp)
        self.X = temp[:, :self.X.shape[1]]
        self.Y = temp[:, self.X.shape[1]:]

    def _predict_prob(self,x):
        output = x
        for i in range(self.n_layers-1):
            output  = output@ self.thetas[i]
            output  += self.bias[i]
            if(self.activation_fun=='relu' and i != self.n_layers-1):
                output = self.relu(output)
            else :
                output  = self.sigmoid(output)
        return output
    
    def score(self, X, Y) :
        out = self._predict_prob(X)
        cor = np.sum(np.argmax(out, axis= 1) == np.argmax(Y, axis =1))
        return cor/Y.shape[0]

    def cal_error(self, j):
        error = self.activations[self.n_layers-1]-self.Y[j:j+self.batch_size, :]
        error = np.sum(error**2)/(2 * self.batch_size)
        return error
  
    def sigmoid(self,A):
        return 1 /(1+ np.exp(-A))
    
    def relu(self, A):
        return np.maximum(0, A)

    def initialize(self):
        self.layers = [self.n_features]+self.hidden_layers+[self.n_classes]
        self.thetas = [np.random.normal(0,0.1,(self.layers[i], self.layers[i+1]) ) for i in range(len(self.layers)-1)]
        self.activations = [np.zeros((self.batch_size, i)) for i in self.layers]
        self.bias = [np.random.normal(0, 0.1, self.layers[i]) for i  in range(1, len(self.layers))]
        self.delta = [np.zeros((self.batch_size,self.layers[j])) for j in range(1, len(self.layers))]
    
    def compute_activations(self , j):
        self.activations[0]= self.X[j:min(j+self.batch_size,self.X.shape[0])]
        for i in range(1,self.n_layers):
            self.activations[i] = self.activations[i-1]@self.thetas[i-1]  + self.bias[i-1]
            if(self.activation_fun=='relu' and i != self.n_layers-1) :
                self.activations[i] = self.relu(self.activations[i])
            else :
                self.activations[i] = self.sigmoid(self.activations[i])

    def back_propagation(self, j):
        O = self.activations[-1]
        Y = self.Y[j:min(j+self.batch_size,self.Y.shape[0])]
        self.delta[-1] = (Y-O)*O*(1-O)
        for i in range(self.n_layers-2, 0,  -1):
            if self.activation_fun == 'relu':
                self.delta[i-1] = (self.delta[i] @ self.thetas[i].T) * np.int64(self.activations[i]>0) 
            else :
                O = self.activations[i]
                self.delta[i-1]= (self.delta[i] @ self.thetas[i].T) * O *(1-O)

    def update_parameters(self):
        for i in range(self.n_layers-2, -1, -1):
            self.thetas[i] += self.alpha *(self.activations[i].T @ self.delta[i])/self.batch_size
            self.bias[i] += self.alpha * np.sum(self.delta[i], axis=0)/self.batch_size

    def fit(self, X, Y):
        self.X =X
        self.Y= Y
        self.initialize()
        prev_error =  1000
        error =0
        count=0
        itr_in_epoch = math.ceil(self.X.shape[0]/self.batch_size)
        initial_alpha =  self.alpha
        for i in range(1,self.max_iter):
            self.shuffle_data()
            if self.learning == 'adaptive' :
                self.alpha = initial_alpha/np.sqrt(i)
            for j in range(0, X.shape[0], self.batch_size):
                self.compute_activations(j)
                self.back_propagation(j)
                self.update_parameters()
                error += self.cal_error(j)
            error /= itr_in_epoch
            if(abs(prev_error- error) < self.stop ) :
                count += 1
                if count ==2 :
                    print('no of epochs', i)
                    return   
            else :
                count=0
            prev_error = error
            error =0         
        print('no of epoch ', self.max_iter)
        print('Done')