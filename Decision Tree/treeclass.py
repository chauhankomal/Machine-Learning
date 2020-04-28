import numpy as np
import math
import pandas as pd
from sklearn import tree
from tqdm import tqdm, trange
import time
import pickle
from xclib.data import data_utils
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV 
from sklearn.metrics import make_scorer
from joblib import Parallel,delayed
from matplotlib import pyplot as plt
from collections import deque


class Node:
    def __init__(self,total_samples,class_predicted,thr, feature, IG):
        self.left= None
        self.right = None
        self.total_samples= total_samples
        self.class_predicted = class_predicted 
        self.threshold = thr
        self.feature= feature
        self.IG = IG

class Decision_tree_classifier:   
    def __init__(self, gtype='dfs'):
        self.nodes=0
        self.test_acc=[]
        self.val_acc=[]
        self.train_acc =[]
        self. no_nodes=[]
        self.gtype= gtype
  
    def entropy(self,y):
        p_y = np.array([np.sum([y==i])/y.shape[0] for i in self.classes])
        return np.sum([ -i* math.log(i) for i in p_y if  i != 0])          
    
    def update_acc(self,nodes, train, test, val):
        nodes.append(self.nodes)
        test.append(self.score(self.test_y,self.predict(self.test_x)))
        train.append(self.score(self.train_y,self.predict(self.train_x)))
        val.append(self.score(self.val_y,self.predict(self.val_x)))

    def find_best_attribute(self, train_x, train_y):
        y=train_y.reshape(-1)
        best_attr =-1
        best_attr_med=-1
        best_IG =-1
        H_Y = self.entropy(y)   
        for i in range(train_x.shape[1]):
            X = train_x[:,i]
            if len(set(X)) ==1:
                continue
            flag = train_x.shape[0]%2
            if(flag==0):
                threshold = np.median(sorted(X)[(X.shape[0]-1)//2])
            else :
                threshold = np.median(X)
            ind = (X <= threshold)   
            if(np.sum(ind)==0 or np.sum(~ind)==0):
                continue
            y_lt_thr = y[ind] 
            y_gt_thr = y[~ind]
            w =(y_lt_thr.shape[0]/y.shape[0])
            left_entropy = self.entropy(y_lt_thr)
            right_entropy = self.entropy(y_gt_thr)
            H_Y_X =  w*left_entropy + (1-w)* right_entropy
            IG = H_Y - H_Y_X
            if(IG > best_IG and IG >= 0):
                best_attr = i
                best_IG =  IG
                best_attr_med = threshold
        return best_attr, best_attr_med, best_IG
    
    def grow_tree(self, root, X, Y):
        if(len(list(set(Y.reshape(-1))))==1) :
            root.total_samples = Y.shape[0]
            root.class_predicted = Y[0][0]
            if(self.nodes%100 == 0):
                self.update_acc(self.no_nodes, self.train_acc, self.test_acc, self.val_acc)
            return 
        class_count = [np.sum(Y==i) for i  in self.classes]
        pred_class= self.classes[np.argmax(class_count)]
        samples = np.sum(class_count)
        root.total_samples = samples
        root.class_predicted = pred_class
        if(self.nodes%100 ==0):
            self.update_acc(self.no_nodes, self.train_acc, self.test_acc, self.val_acc)
        attr, med, IG= self.find_best_attribute(X,Y)
        if IG == -1 :
            return 
        root.threshold =med
        root.feature  =  attr
        root.IG = IG
        ind = X[:,attr] <= med 
        root.left= Node(-1,-1,-1,-1,0)
        self.nodes +=1
        self.grow_tree(root.left,X[ind], Y[ind])
        root.right= Node(-1,-1,-1,-1,0)
        self.nodes +=1
        self.grow_tree(root.right, X[~ind], Y[~ind])

    def grow_tree_level(self, X, Y):
        self.update_acc(self.no_nodes, self.train_acc, self.test_acc, self.val_acc)
        prev_nodes=0
        self.q = deque()
        self.q.append((self.tree, X, Y))
        while self.q :
            curr_node, X, Y = self.q.popleft()
            self.nodes +=1
            if (len(list(set(Y.reshape(-1))))==1):
                curr_node.total_samples = Y.shape[0]
                curr_node.class_predicted = Y[0][0]
                if(self.nodes -prev_nodes >=100):
                    self.update_acc(self.no_nodes, self.train_acc, self.test_acc, self.val_acc)
                    prev_nodes =self.nodes
                continue    
            class_count = [np.sum(Y==i) for i  in self.classes]
            pred_class= self.classes[np.argmax(class_count)]
            samples = np.sum(class_count)
            curr_node.total_samples = samples
            curr_node.class_predicted = pred_class
            if(self.nodes -prev_nodes >=100):
                self.update_acc(self.no_nodes, self.train_acc, self.test_acc, self.val_acc)
                prev_nodes =self.nodes
            attr, med, IG= self.find_best_attribute(X,Y)      
            if IG ==-1 :
                continue
            curr_node.threshold =med
            curr_node.feature  =  attr
            curr_node.IG = IG
            left_ind = X[:,attr] <= med 
            curr_node.left= Node(-1,-1,-1,-1,0)
            self.q.append((curr_node.left,X[left_ind], Y[left_ind]))
            curr_node.right= Node(-1,-1,-1,-1,0)
            self.q.append((curr_node.right, X[~left_ind], Y[~left_ind]))

    def fit(self, X, Y, test_x, test_y, val_x, val_y):
        self.train_x = X
        self.train_y = Y
        self.test_x = test_x
        self.test_y  = test_y
        self.val_x = val_x
        self.val_y = val_y
        self.c=0
        self.classes= sorted(list(set(Y.reshape(-1))))
        if(self.gtype== 'level') :
            class_count = [np.sum(Y==i) for i  in self.classes]
            pred_class= self.classes[np.argmax(class_count)]
            self.tree = Node(-1,pred_class,-1,-1,0)
            print('growing tree level wise')
            self.grow_tree_level(X, Y)
            self.update_acc(self.no_nodes, self.train_acc, self.test_acc, self.val_acc)
        else :
            self.nodes += 1
            self.tree = Node(-1,-1,-1,-1,0)
            print('growing tree recursively')
            self.grow_tree(self.tree, X, Y) 
            self.update_acc(self.no_nodes, self.train_acc, self.test_acc, self.val_acc)      
        
    def predict1(self,x):
        node = self.tree
        while(node.left and node.left.class_predicted !=-1):
            if(x[node.feature]<=node.threshold):
                node=node.left
            else :
                if node.right== None:
                    return node.class_predicted
                node=node.right
        return node.class_predicted

    def predict(self,inp):
        return np.array([self.predict1(i.reshape(-1)) for i in inp]).ravel()   
   
    def score(self, y_actual, y_pred):
        correct =0
        for i in range(len(y_actual)): 
            if y_actual[i] == y_pred[i]:
                correct +=1
        return (correct/len(y_actual))

    def count_nodes(self, root):
        if root.left == None:
            self.count+=1
            return
        self.count+=1
        self.count_nodes(root.left)
        self.count_nodes(root.right)
        
    def post_pruning(self,root, X, y ):
        correct = sum(y == root.class_predicted)
        if(root.left == None):
            return correct
        ind = X[:,root.feature]<=root.threshold
        corr1=self.post_pruning(root.left, X[ind], y[ind])
        corr2= self.post_pruning(root.right, X[~ind], y[~ind])
        if(correct >= corr1+corr2):
            self.count =0
            self.count_nodes(root)
            self.prune_nodes += (self.count -1)
            self.nodes -= (self.count -1)
            if((self.prune_nodes - self.prev_prune) >= 200):
                self.prev_prune =  self.prune_nodes
                self.update_acc(self.prune_no_nodes, self.prune_train_acc, self.prune_test_acc, self.prune_val_acc)   
            root.left=None
            root.right = None
        else :
            correct = corr1+corr2
        return correct
    
    def prune_tree(self):
        self.prune_test_acc = []
        self.prune_val_acc =  []
        self.prune_train_acc = []
        self.prune_no_nodes=  []
        self.prune_nodes = 0
        self.prev_prune=0
        self.count =  0
        self.update_acc(self.prune_no_nodes, self.prune_train_acc, self.prune_test_acc, self.prune_val_acc)
        a=self.post_pruning(self.tree, self.val_x, self.val_y)
        self.update_acc(self.prune_no_nodes, self.prune_train_acc, self.prune_test_acc, self.prune_val_acc)
