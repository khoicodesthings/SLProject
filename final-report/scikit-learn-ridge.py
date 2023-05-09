import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import RidgeClassifier #Ridge Regression Classifier
import time

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

clf = RidgeClassifier(alpha = 0.5, max_iter=1000)
start = time.time()
ridge = clf.fit(X_train, y_train)

y_pred = ridge.predict(X_val)
end = time.time()

print('Total time:', (end-start)*1000, 'miliseconds')
# Evaluate model accuracy
accuracy = np.sum(y_val == y_pred) / len(y_val)
print('Accuracy:', accuracy)

# Print the confusion matrix
cm = confusion_matrix(y_val, y_pred)
print("Confusion matrix:")
print(cm)

# Extract TP, FP, FN, TN from confusion matrix
TP = cm[1, 1]
FP = cm[0, 1]
FN = cm[1, 0]
TN = cm[0, 0]

# Compute precision and recall
precision = TP / (TP + FP)
recall = TP / (TP + FN)

print('Precision:',precision)
print('Recall:',recall)

# Compute the false positive rate and true positive rate
fpr, tpr, thresholds = roc_curve(y_val, y_pred)

# Compute the area under the ROC curve
roc_auc = auc(fpr, tpr)

print('AUC:', roc_auc)
# 0.96857

# Plot ROC curve
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Ridge Regression ROC Curve')
plt.legend(loc="lower right")
#plt.savefig('scikitridgeROC.png')
plt.show()