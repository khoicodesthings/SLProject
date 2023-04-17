import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

# Scale the features using mean normalization
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

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

tp = true_positives(y_val, y_pred)
fp = false_positives(y_val, y_pred)

precision = tp / (tp + fp)
recall = tp / (tp + np.sum(y_val == 1))

print('Precision:', precision)
print('Recall:', recall)

# Evaluate model accuracy
accuracy = np.sum(y_val == y_pred) / len(y_val)
print('Accuracy:', accuracy)

# Calculate true positive rate and false positive rate for various thresholds
thresholds = np.linspace(0, 1, 100)
tpr = []
fpr = []
for threshold in thresholds:
    y_pred_threshold = np.where(sigmoid(X_val.dot(theta)) >= threshold, 1, 0)
    tp = true_positives(y_val, y_pred_threshold)
    fp = false_positives(y_val, y_pred_threshold)
    tn = np.sum((y_val == 0) & (y_pred_threshold == 0))
    fn = np.sum((y_val == 1) & (y_pred_threshold == 0))
    tpr.append(tp / (tp + fn))
    fpr.append(fp / (fp + tn))

# Plot ROC curve
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.show()