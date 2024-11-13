#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 22:56:37 2023

@author: airishimamura
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 17:26:26 2023

@author: airishimamura
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_curve, roc_auc_score,auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn. metrics import recall_score


import time

class Node:
    def __init__(self, data, target):
        self.left = None
        self.right = None
        self.feature = None
        self.threshold = None
        self.leaf = False
        self.prediction = None
        self.data = data
        self.target = target

class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, data, target):
        # call a recursive function to build the tree
        self.root = self.build_tree(data, target, depth=0)

    def build_tree(self, data, target, depth):
        num_samples, num_features = data.shape
        num_pos = (target == 1).sum()
        num_neg = (target == 0).sum()

         # base cases: check stopping criteria
        if depth == self.max_depth or num_samples <= 1 or num_pos == 0 or num_neg == 0:
            leaf = Node(data, target)
            leaf.leaf = True
            leaf.prediction = 1 if num_pos > num_neg else 0
            return leaf

        # find best feature and threshold to split on
        best_feature = 0
        best_threshold = 0.0
        best_gini = 1.0

        for feature in range(num_features):
            # get all the possible thresholds
            thresholds = np.unique(data[:, feature])
            for threshold in thresholds:
                left_indices = data[:, feature] <= threshold
                right_indices = data[:, feature] > threshold

                if left_indices.sum() == 0 or right_indices.sum() == 0:
                    continue

                left_target = target[left_indices]
                right_target = target[right_indices]
                
                # Calculate the Gini index for a split dataset
                gini_left = 1.0 - ((left_target == 1).sum() / left_target.shape[0])**2 - ((left_target == 0).sum() / left_target.shape[0])**2
                gini_right = 1.0 - ((right_target == 1).sum() / right_target.shape[0])**2 - ((right_target == 0).sum() / right_target.shape[0])**2

                weighted_gini = (left_target.shape[0] / num_samples) * gini_left + (right_target.shape[0] / num_samples) * gini_right
                
                # find the best split 
                if weighted_gini < best_gini:
                    best_feature = feature
                    best_threshold = threshold
                    best_gini = weighted_gini

        # split on best feature and threshold
        left_indices = data[:, best_feature] <= best_threshold
        right_indices = data[:, best_feature] > best_threshold

        # grow subtrees(child nodes)
        node = Node(data, target)
        node.feature = best_feature
        node.threshold = best_threshold
        node.left = self.build_tree(data[left_indices], target[left_indices], depth+1)
        node.right = self.build_tree(data[right_indices], target[right_indices], depth+1)

        return node

    def predict(self, data):
        predictions = []
        for i in range(data.shape[0]):
            node = self.root
            while not node.leaf:
                if data[i, node.feature] <= node.threshold:
                    node = node.left
                else:
                    node = node.right
            predictions.append(node.prediction)
        return np.array(predictions)

class RandomForest:
    def __init__(self, n_estimators=10, max_depth=None, max_features=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.trees = []

    def fit(self, data, target):
        for i in range(self.n_estimators):
            # Sample random subset of features
            if self.max_features is not None:
                feature_indices = np.random.choice(data.shape[1], self.max_features, replace=False)
                data_subset = data[:, feature_indices]
            else:
                data_subset = data

            # Sample random subset of data <- same as the boostsample fubc
            #indices = np.random.choice(data.shape[0], data.shape[0], replace=True)
            #data_sampled = data_subset[indices]
            #target_sampled = target[indices]
            
            data_sampled, target_sampled = self.bootstrapSample(data, target)

            # Train decision tree on the sampled data
            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(data_sampled, target_sampled)

            # Add the trained tree to the list of trees
            self.trees.append(tree)
            
    def bootstrapSample(self, data, target):
        num_samples = data.shape[0]
        # choose a number of samples randomly from the dataset and replace it then select some samples again 
        idxs = np.random.choice(num_samples, num_samples, replace=True)
        return data[idxs], target[idxs]
    
    
    def predict(self, data):
        predictions = np.zeros((data.shape[0], len(self.trees)))

        # Make predictions with each tree in the forest
        for i, tree in enumerate(self.trees):
            predictions[:, i] = tree.predict(data)

        # Take the majority vote as the final prediction
        final_predictions = np.zeros(data.shape[0])
        for i in range(data.shape[0]):
            final_predictions[i] = np.bincount(predictions[i, :].astype(int)).argmax()

        return final_predictions.astype(int)




# Load dataset from final.csv file
df = pd.read_csv('final.csv')

df['key'] = df['key'].astype('category')
df['mode'] = df['mode'].astype('category')

# Shuffle the dataframe
df = df.sample(frac=1, random_state=42)

# Extract song names
song_names = df['name']

# Remove duplicates from the dataset
df = df.drop_duplicates()

# Split data into features and target variable
X = df.iloc[:, 1:-1].values
y = df.iloc[:, -1].values

# define a scaling function for numpy arrays
def scale_features(X):
    for i in range(X.shape[1]):
        col = X[:, i]
        if np.issubdtype(col.dtype, np.number):  # check if the column is numeric
            col = col.astype(float)
            X[:, i] = (col - col.mean()) / col.std()  # scale the column

# apply the scaling function to X
scale_features(X)

# Set random seed for reproducibility
np.random.seed(5033)

# Split data into train and validation sets
idx = np.random.permutation(len(X))
train_idx, val_idx = idx[:int(0.8*len(X))], idx[int(0.8*len(X)):]

X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]

# --------------------------------------------------------------


rf = RandomForest(n_estimators=30,max_depth=10)
start = time.time()
rf.fit(X_train, y_train)
# estimate the model 
y_pred = rf.predict(X_val)
end = time.time()
print("The time of execution of randomforest is :", (end-start) * 10**3, "ms")


# accuracy  = []
# precision = []
# recall    = []
# y_pred_list = [] 



# t_range = range(10,101,10)
# for t in t_range:
#     rf = RandomForest(n_estimators=t,max_depth=10)
#     rf.fit(X_train, y_train)
    
#     # estimate the model 
#     y_pred = rf.predict(X_val)
#     y_pred_list.append(y_pred)
#     accuracy.append(accuracy_score(y_val,y_pred))
#     precision.append(precision_score(y_val,y_pred, average="macro"))
#     recall.append(recall_score(y_val,y_pred, average="macro"))
#     print("finish: ", t)
    
# f = plt.figure(1)
# # plot
# plt.plot(t_range, accuracy,  label="accuracy")
# plt.plot(t_range, precision, label="precision")
# plt.plot(t_range, recall,    label="recall")
# plt.legend(loc="best")
# plt.xlabel('number of trees')
# plt.ylabel('score')
# plt.title('Randomforest')
# plt.savefig('randomforest_scores.png')
# f.show()

# # print results 
# max_accuracy = max(accuracy)
# index = accuracy.index(max_accuracy)
# best_t_range = t_range[index]
# best_precision = precision[index]
# best_recall = recall[index]
# print("when n_tree =" +str(best_t_range)+"\nmax accuracy is "+str(max_accuracy) + "\nprecision is "+str(best_precision) + "\nrecall is " + str(best_recall)) 

# # # Evaluate the performance of the model on the validation set
# # Print the confusion matrix
# cm = confusion_matrix(y_val, y_pred)
# print("Confusion matrix:")
# print(cm)

# # Extract TP, FP, FN, TN from confusion matrix
# TP = cm[1, 1]
# FP = cm[0, 1]
# FN = cm[1, 0]
# TN = cm[0, 0]

# # Compute the false positive rate and true positive rate
# fpr, tpr, thresholds = roc_curve(y_val, y_pred)

# # Compute the area under the ROC curve
# roc_auc = auc(fpr, tpr)

# # Plot ROC curve
# s = plt.figure(2)
# plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
# plt.plot([0, 1], [0, 1], 'k--')
# plt.xlim([-0.01, 1.01])
# plt.ylim([-0.01, 1.01])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Random Forest ROC Curve')
# plt.legend(loc="lower right")
# plt.savefig('random-forest-ROC.png')
# s.show()


# # # plot roc curve 
# #s = plt.figure(2)
# # test_fpr, test_tpr, te_thresholds = roc_curve(y_val, y_pred_list[index])
# # plt.figure(figsize=(6,5))
# # plt.plot([0, 1], [0, 1], "k--")
# # plt.plot(test_fpr, test_tpr,linewidth=3)

# # # # Plot ROC curve
# # plt.xlabel("False Positive Rate")
# # plt.ylabel("True Positive Rate")
# # plt.title('Random Forest ROC Curve')
# # plt.legend(loc="lower right")
# # plt.savefig('randomforest-ROC.png')
# # s.show()
