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
import copy
import sys
from treeclass import Decision_tree_classifier



train_x_path = sys.argv[1]
train_y_path = sys.argv[2]
test_x_path = sys.argv[3]
test_y_path = sys.argv[4]
val_x_path = sys.argv[5]
val_y_path = sys.argv[6]

train_x = data_utils.read_sparse_file(train_x_path, force_header=True).toarray()
test_x= data_utils.read_sparse_file(test_x_path, force_header=True).toarray()
val_x = data_utils.read_sparse_file(val_x_path, force_header=True).toarray()
train_y = np.array(pd.read_csv(train_y_path, header = None))
test_y = np.array(pd.read_csv(test_y_path, header = None))
val_y = np.array(pd.read_csv(val_y_path, header = None))

def plot_node_acc(test_score,train_acc, val_score,x, var_param_name ):
    ax= plt.figure(figsize=(12,7))
    plt.plot(x, train_acc,color='green')
    plt.plot(x,test_score)
    plt.plot(x,val_score, color='r')
    plt.legend(['Train Accuracy', 'Test Accuracy', 'Validation Accuracy'])
    plt.xlabel(var_param_name)
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Number of nodes')
    plt.show()


def plot_score(test_score, val_score,x, var_param_name, title ):
  ax= plt.figure(figsize=(14,6))
  plt.plot(x,test_score)
  plt.plot(x,val_score)
  plt.legend(['Test set', 'Validation set'])
  plt.xlabel(var_param_name)
  plt.ylabel('Accuracy')
  plt.title(title)
  plt.xticks(x)
  plt.savefig(var_param_name, dpi=300)
  plt.show()


print('Training decision tree classifier...')
d = Decision_tree_classifier('dfs')
start = time.time()
d.fit(train_x, train_y, test_x, test_y, val_x, val_y)
print('Time taken in training',(time.time()-start)/60, ' min')
print('No. of nodes in tree',d.nodes)


val_y_pred= d.predict(val_x)
test_y_pred= d.predict(test_x)
train_y_pred= d.predict(train_x)
print('validation accuracy using decision tree', d.score(val_y, val_y_pred)*100)
print('test accuracy using decision tree', d.score(test_y, test_y_pred)*100)
print('training accuracy using decision tree', d.score(train_y, train_y_pred)*100)

plot_node_acc(d.test_acc, d.train_acc, d.val_acc,d.no_nodes, 'Number of nodes')
t = copy.deepcopy(d);
t.prune_tree()

print('Number of nodes  after prunning', t.nodes)

train_y_pred=np.array(t.predict(train_x)).reshape(-1)
acc_train=t.score(train_y,train_y_pred)*100
print('Accuracy on train data after pruning', acc_train)
val_y_pred=np.array(t.predict(val_x)).reshape(-1)
acc_val=t.score(val_y,val_y_pred)*100
print('Accuracy on validation set after pruning', acc_val)
test_y_pred=np.array(t.predict(test_x)).reshape(-1)
acc_test=t.score(test_y,test_y_pred)*100
print('Accuracy on test set after pruning', acc_test)

plot_node_acc(t.prune_test_acc, t.prune_train_acc, t.prune_val_acc,t.prune_no_nodes, 'Number of nodes')

# (c) Random Forest

param_grid = [
  {'n_estimators': [50, 150, 250, 350, 450], 'max_features': [0.1, 0.3, 0.5, 0.7,0.9], 
   'min_samples_split':[2,4,6,8,10]}]

def scorer(rclf, y_pred, y_actual):
  return rclf.oob_score_

s = time.time()
rand_clf =RandomForestClassifier(oob_score = True,n_jobs=-2)
clf = GridSearchCV(rand_clf, param_grid, scoring= scorer, cv = 5,verbose=1)
clf.fit(train_x,train_y.ravel())
print('Time taken',(time.time()-s)/60)


grid_clf_file= open('grid_clf', 'wb')
pickle.dump(clf, grid_clf_file)
grid_clf_file= open('grid_clf', 'rb')
grid_clf = pickle.load(grid_clf_file)
best_param = grid_clf.best_params_
print(best_param)

# accuracy with best estimator

train_acc= grid_clf.best_estimator_.score(train_x, train_y)
test_acc =grid_clf.best_estimator_.score(test_x, test_y)
val_acc = grid_clf.best_estimator_.score(val_x, val_y)
oob_acc = grid_clf.best_estimator_.oob_score_
print('Training set accuracy using optimal parameters', train_acc*100)
print('out of bag accuracy',oob_acc*100)
print('Test set accuracy using optimal parameters', test_acc*100)
print('Validation accuracy using optimal parameters', val_acc*100)



# (d) Random Forests - Parameter Sensitivity Analysis:


n_estimators = [10,30,50, 150, 250, 350, 450, 550, 650, 750, 850, 950]
est_acc=[]
s = time.time()
for j in range(5):
    c =0
    for i in tqdm(n_estimators):
        clf = RandomForestClassifier(n_estimators=i, max_features=best_param['max_features'],min_samples_split=best_param['min_samples_split'], n_jobs=-2,oob_score=True )
        clf.fit(train_x, train_y.ravel())
        if(j>0):
            est_acc[c]+=np.array([clf.score(test_x, test_y)*100, clf.score(val_x, val_y)*100])
        else :
            est_acc.append(np.array([clf.score(test_x, test_y)*100, clf.score(val_x, val_y)*100]))
        c+=1
est_acc = np.array(est_acc)/5
print('\nTime taken  ',(time.time()-s)/60, ' min')


max_features = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
features_acc=[]
s = time.time()
for j in range(5):
    c =0
    for i in tqdm(max_features):
        clf = RandomForestClassifier(n_estimators=best_param['n_estimators'], max_features=i,min_samples_split=best_param['min_samples_split'], n_jobs=-2)
        clf.fit(train_x, train_y.ravel())
        if(j>0):
            features_acc[c]+=np.array([clf.score(test_x, test_y)*100, clf.score(val_x, val_y)*100])
        else :
            features_acc.append(np.array([clf.score(test_x, test_y)*100, clf.score(val_x, val_y)*100]))
        c+=1
features_acc = np.array(features_acc)/5       
print('Time taken  ',(time.time()-s)/60, ' sec')


min_samples_split = [2, 4, 6, 8 ,10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
samples_acc =[]
for j in range(5):
    c =0
    for  i in min_samples_split:
        clf = RandomForestClassifier(n_estimators=best_param['n_estimators'], max_features=best_param['max_features'],min_samples_split=i, n_jobs=-2 )
        clf.fit(train_x, train_y.ravel())
        if(j>0):
            samples_acc[c]+=np.array([clf.score(test_x, test_y)*100, clf.score(val_x, val_y)*100])
        else :
            samples_acc.append(np.array([clf.score(test_x, test_y)*100, clf.score(val_x, val_y)*100]))
        c+=1
samples_acc = np.array(samples_acc)/5       
print('\nTime taken  ',(time.time()-s)/60, ' sec')


test_score= np.array(est_acc)[:,0]
val_score = np.array(est_acc)[:,1]
title = 'n_estimators vs Accuracy'
print(test_score)
print(val_score)
plot_score(test_score, val_score,n_estimators, 'n_estimators',title)

test_score= np.array(features_acc)[:,0]
val_score = np.array(features_acc)[:,1]
title = 'max_features vs Accuracy'
print(test_score)
print(val_score)
plot_score(test_score, val_score,max_features, 'max_features', title)

test_score= np.array(samples_acc)[:,0]
val_score = np.array(samples_acc)[:,1]
title = 'min_samples_split vs Accuracy'
print(test_score)
print(val_score)
plot_score(test_score, val_score,min_samples_split, 'min_samples_split',title)

