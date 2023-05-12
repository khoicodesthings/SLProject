#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  5 02:41:41 2023

@author: airishimamura
"""
import time 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn. metrics import recall_score
from sklearn.metrics import confusion_matrix

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
#df = df.drop(columns=['name', 'key', 'mode'])

# Split data into features and target variable
X = df.iloc[:, 1:-1].values
y = df.iloc[:, -1].values

# Scale the features using mean normalization
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Set random seed for reproducibility
np.random.seed(5033)

# Split data into train and validation sets
idx = np.random.permutation(len(X))
train_idx, val_idx = idx[:int(0.8*len(X))], idx[int(0.8*len(X)):]

x_train, y_train = X[train_idx], y[train_idx]
x_val, y_val = X[val_idx], y[val_idx]




# knn = KNeighborsClassifier(n_neighbors=10)
# start = time.time()
# knn.fit(x_train,y_train)
# y_pred = knn.predict(x_val)
# end = time.time()
# print("The time of execution of knn is :", (end-start) * 10**3, "ms")

# # evaluate the model
# print("accuracy is: " + str(accuracy_score(y_val,y_pred)))
# print("precision is: " + str(precision_score(y_val,y_pred, average="macro")))
# print("recall is: " +  str(recall_score(y_val,y_pred, average="macro")))



accuracy  = []
precision = []
recall    = []
y_pred_list = [] 

k_range = range(1,100)

for k in k_range:

    knn = KNeighborsClassifier(n_neighbors=k)
    
    knn.fit(x_train,y_train)
    
    # evaluate the model
    y_pred = knn.predict(x_val)
    y_pred_list.append(y_pred)
    accuracy.append(accuracy_score(y_val,y_pred))
    precision.append(precision_score(y_val,y_pred, average="macro"))
    recall.append(recall_score(y_val,y_pred, average="macro"))
 
    
f = plt.figure(1)
# plot
plt.plot(k_range, accuracy,  label="accuracy")
plt.plot(k_range, precision, label="precision")
plt.plot(k_range, recall,    label="recall")
plt.legend(loc="best")
plt.xlabel('number of neighbors')
plt.ylabel('score')
plt.title('knn')
plt.savefig('knn_scores.png')
f.show()

# print results 
max_accuracy = max(accuracy)
index = accuracy.index(max_accuracy)
best_k_range = k_range[index]
best_precision = precision[index]
best_recall = recall[index]
print("when k=" +str(best_k_range)+"\nmax accuracy is "+str(max_accuracy) + "\nprecision is "+str(best_precision) + "\nrecall is " + str(best_recall)) 


# plot roc curve 
from sklearn.metrics import roc_curve, roc_auc_score,auc
# test_fpr, test_tpr, te_thresholds = roc_curve(y_val, y_pred_list[index])
# plt.plot(test_fpr, test_tpr,linewidth=3)

cm = confusion_matrix(y_val, y_pred)
print("Confusion matrix:")
print(cm)

# Extract TP, FP, FN, TN from confusion matrix
TP = cm[1, 1]
FP = cm[0, 1]
FN = cm[1, 0]
TN = cm[0, 0]

# Compute the false positive rate and true positive rate
fpr, tpr, thresholds = roc_curve(y_val, y_pred)

# Compute the area under the ROC curve
roc_auc = auc(fpr, tpr)

# Plot ROC curve
s = plt.figure(2)
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC curve: Knn")
plt.legend(loc="lower right")
plt.savefig('knn-ROC.png')
s.show()





