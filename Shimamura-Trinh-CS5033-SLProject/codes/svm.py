#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  5 09:16:16 2023

@author: airishimamura
"""
import time 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics 
from sklearn.metrics  import confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn. metrics import recall_score
from sklearn.metrics import roc_curve, roc_auc_score,auc

# Load dataset from final.csv file
df = pd.read_csv('final.csv')
# change two variables to categorical variables 
df['key'] = df['key'].astype('category')
df['mode'] = df['mode'].astype('category')

# Shuffle the dataframe
df = df.sample(frac=1, random_state=42)

# Extract song names
song_names = df['name']

# Remove duplicates from the dataset
df = df.drop_duplicates()
# remove categorical variables 
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

# make a model 
svm_model = SVC()

start = time.time()
svm_model.fit(x_train, y_train)
y_pred = svm_model.predict(x_val)
end = time.time()
print("The time of execution of svm is :", (end-start) * 10**3, "ms")

# evaluate the model 
#result = svm_model.score(x_val, y_val)
#print("Result score: ", result) # same as acuracy score  
print("Accuracy: " + str(accuracy_score(y_val,y_pred)))
print("Precision: " + str(precision_score(y_val,y_pred, average="macro")))
print("Recall: " + str(recall_score(y_val,y_pred, average="macro")))

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
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([-0.01, 1.01])

test_fpr, test_tpr, te_thresholds = roc_curve(y_val, y_pred)
#plt.plot(test_fpr, test_tpr,linewidth=3)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC curve: SVM")
plt.legend(loc="lower right")
plt.savefig('svm-ROC.png')
plt.show()



