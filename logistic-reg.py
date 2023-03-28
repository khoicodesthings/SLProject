import pandas as pd
import numpy as np

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

# Scale the features using mean normalization
x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)

# Split data into training and testing sets
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Split data into training and testing sets
np.random.seed(5033)
indices = np.random.permutation(len(df))
train_indices, test_indices = indices[:int(0.8*len(df))], indices[int(0.8*len(df)):]

x_train, y_train = x.iloc[train_indices], y.iloc[train_indices]
x_test, y_test = x.iloc[test_indices], y.iloc[test_indices]

# Define sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Define cost function
def cost_function(x, y, theta):
    m = len(y)
    h = sigmoid(x.dot(theta))
    eps = 1e-15 # small constant value to avoid taking the logarithm of zero
    J = -1/m * (y.T.dot(np.log(h + eps)) + (1-y).T.dot(np.log(1-h + eps)))
    return J

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
    J_history = []
    for i in range(num_iters):
        cost = cost_function(x, y, theta)
        grad = gradient(x, y, theta)
        theta = theta - alpha * grad
        J_history.append(cost)
    return theta, J_history

# Train logistic regression model
alpha = 0.01
num_iters = 1000
theta, J_history = logistic_regression(x_train, y_train, alpha, num_iters)

# Make predictions
y_pred = np.round(sigmoid(x_test.dot(theta)))

# Extract song names from index
song_names_test = song_names[x_test.index]

# Save predictions to CSV file
#results = pd.DataFrame({'song_name': song_names_test, 'preference_pred': y_pred})
#results = results[['song_name', 'preference_pred']]
#results.to_csv('song_predictions.csv', index=False)

# Extract correct preferences from the test set
y_true = y_test.values

# Save predictions and true preferences to CSV file
results = pd.DataFrame({'song_name': song_names_test, 'preference_true': y_true, 'preference_pred': y_pred})
results = results[['song_name', 'preference_true', 'preference_pred']]
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
accuracy = np.sum(y_test == y_pred) / len(y_test)
print('Accuracy:', accuracy)
