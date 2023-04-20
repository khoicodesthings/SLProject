import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc

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

# Define sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Define loss function
def loss(x, y, theta):
    m = len(y)
    h = sigmoid(x.dot(theta))
    eps = 1e-15 # small constant value to avoid taking the logarithm of zero
    j = -1/m * (y.T.dot(np.log(h + eps)) + (1-y).T.dot(np.log(1-h + eps)))
    return j

# Define gradient function
def gradient(x, y, theta):
    m = len(y)
    h = sigmoid(x.dot(theta))
    grad = 1/m * x.T.dot(h-y)
    return grad

# Define logistic regression function
def logistic_regression(x, y, alpha, num_iters):
    m, n = x.shape
    theta = np.zeros(n)
    j_history = []
    for i in range(num_iters):
        cost = loss(x, y, theta)
        grad = gradient(x, y, theta)
        theta = theta - alpha * grad
        j_history.append(cost)
    return theta, j_history

# Calculate precision and recall
def true_positives(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    return tp

def false_positives(y_true, y_pred):
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return fp

# Train logistic regression model
alpha = 1
num_iters = 1000
theta, j_history = logistic_regression(X_train, y_train, alpha, num_iters)

# Make predictions
y_pred = np.round(sigmoid(X_val.dot(theta)))

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

# Plot ROC curve
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression ROC Curve')
plt.legend(loc="lower right")
plt.savefig('logistic-regressionROC.png')
plt.show()