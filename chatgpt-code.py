import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Read in CSV file with song features and binary preference
df = pd.read_csv('final.csv')

# Shuffle the dataframe
df = df.sample(frac=1, random_state=42)

# Extract song names
song_names = df['name']

# Remove duplicates
df = df.drop_duplicates()

# Split data into features and target variable
x = df.drop(columns=['name', 'preference'])
y = df['preference']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Define sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Define cost function
def cost_function(X, y, theta):
    m = len(y)
    h = sigmoid(X.dot(theta))
    eps = 1e-15 # small constant value to avoid taking the logarithm of zero
    J = -1/m * (y.T.dot(np.log(h + eps)) + (1-y).T.dot(np.log(1-h + eps)))
    return J

# Define gradient function
def gradient(X, y, theta):
    m = len(y)
    h = sigmoid(X.dot(theta))
    grad = 1/m * X.T.dot(h-y)
    return grad

# Define logistic regression function
def logistic_regression(X, y, alpha, num_iters):
    m, n = X.shape
    theta = np.zeros(n)
    J_history = []
    for i in range(num_iters):
        cost = cost_function(X, y, theta)
        grad = gradient(X, y, theta)
        theta = theta - alpha * grad
        J_history.append(cost)
    return theta, J_history

# Train logistic regression model
alpha = 0.01
num_iters = 1000
theta, J_history = logistic_regression(X_train, y_train, alpha, num_iters)

# Make predictions
y_pred = np.round(sigmoid(X_test.dot(theta)))

# Extract song names from index
song_names_test = song_names[X_test.index]

# Save predictions to CSV file
results = pd.DataFrame({'song_name': song_names_test, 'preference_pred': y_pred})
results = results[['song_name', 'preference_pred']]
results.to_csv('song_predictions.csv', index=False)

# Calculate precision and recall
def true_positives(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    return tp

def false_positives(y_true, y_pred):
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return fp

tp = true_positives(y_test, y_pred)
fp = false_positives(y_test, y_pred)

precision = tp / (tp + fp)
recall = tp / (tp + np.sum(y_test == 1))

print('Precision:', precision)
print('Recall:', recall)

# Evaluate model accuracy
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)
