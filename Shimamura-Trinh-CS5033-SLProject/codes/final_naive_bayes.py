#!/usr/bin/env ptargetthon3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 10:16:33 2023

@author: airishimamura
"""
import time 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


class NaiveBayes:

    def fit(self, X, target):
        num_samples, num_features = X.shape
        self.classes = np.unique(target)
        total_classes = len(self.classes)

        # get mean, variance, and prior for each class
        self.mean = np.zeros((total_classes, num_features), dtype=np.float64)
        self.variable = np.zeros((total_classes, num_features), dtype=np.float64)
        self.priors = np.zeros(total_classes, dtype=np.float64)

        for idx, class_label in enumerate(self.classes):
            X_c = X[target == class_label]
            self.mean[idx, :] = X_c.mean(axis=0)
            self.variable[idx, :] = X_c.var(axis=0)
            self.priors[idx] = X_c.shape[0] / float(num_samples)

    def predict(self, X):
        target_pred = [self.cal_post_prob(x) for x in X]
        return np.array(target_pred)
    
    # calculate posteriors 
    def cal_post_prob(self,x):
        posteriors = []

        # calculate posterior probability for each class
        for idx, c in enumerate(self.classes):
            prior = np.log(self.priors[idx])
            posterior = np.sum(np.log(self.get_pdf(idx, x)))
            posterior = posterior + prior
            posteriors.append(posterior)

        # return class with the highest posterior
        return self.classes[np.argmax(posteriors)]
    
    # probability density func
    def get_pdf(self, class_idx, x):
        mean = self.mean[class_idx]
        variable = self.variable[class_idx]
        numerator = np.exp(-((x - mean) ** 2) / (2 * variable))
        denominator = np.sqrt(2 * np.pi * variable)
        return numerator / denominator 
    

# Calculate precision and recall
def true_positives(target_true, target_pred):
    tp = np.sum((target_true == 1) & (target_pred == 1))
    return tp

def false_positives(target_true, target_pred):
    fp = np.sum((target_true == 0) & (target_pred == 1))
    return fp


# --------------------------------------------------------------------  

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
target = df.iloc[:, -1].values

# Scale the features using mean normalization
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Set random seed for reproducibilittarget
np.random.seed(5033)

# Split data into train and validation sets
idx = np.random.permutation(len(X))
train_idx, val_idx = idx[:int(0.8*len(X))], idx[int(0.8*len(X)):]

X_train, target_train = X[train_idx], target[train_idx]
X_val, target_val = X[val_idx], target[val_idx]

# make a model 
nb = NaiveBayes()
start = time.time()
nb.fit(X_train, target_train)
target_pred = nb.predict(X_val)
end = time.time()
print("The time of execution of naive bayes is :", (end-start) * 10**3, "ms")

## --------------------------------------------------------------------------------
# Evaluate the performance of the model on the validation set
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve, roc_auc_score,auc

print("Accuracy: " + str(accuracy_score(target_val,target_pred)))
print("Precision: " + str(precision_score(target_val,target_pred, average="macro")))
print("Recall: " + str(recall_score(target_val,target_pred, average="macro")))

cm = confusion_matrix(target_val, target_pred)
print("Confusion matrix:")
print(cm)

# Extract TP, FP, FN, TN from confusion matrix
TP = cm[1, 1]
FP = cm[0, 1]
FN = cm[1, 0]
TN = cm[0, 0]

# Compute the false positive rate and true positive rate
fpr, tpr, thresholds = roc_curve(target_val, target_pred)

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
plt.title('ROC curve: Naive Bayes')
plt.legend(loc="lower right")
plt.savefig('naive-bayes-ROC.png')
plt.show()


