import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
alpha = 0.01
num_iters = 1000
theta, j_history = logistic_regression(x_train, y_train, alpha, num_iters)

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

tp = true_positives(y_test, y_pred)
fp = false_positives(y_test, y_pred)

precision = tp / (tp + fp)
recall = tp / (tp + np.sum(y_test == 1))

print('Precision:', precision)
print('Recall:', recall)

# Evaluate model accuracy
accuracy = np.sum(y_test == y_pred) / len(y_test)
print('Accuracy:', accuracy)

# Calculate true positive rate and false positive rate for various thresholds
thresholds = np.linspace(0, 1, 100)
tpr = []
fpr = []
for threshold in thresholds:
    y_pred_threshold = np.where(sigmoid(x_test.dot(theta)) >= threshold, 1, 0)
    tp = true_positives(y_test, y_pred_threshold)
    fp = false_positives(y_test, y_pred_threshold)
    tn = np.sum((y_test == 0) & (y_pred_threshold == 0))
    fn = np.sum((y_test == 1) & (y_pred_threshold == 0))
    tpr.append(tp / (tp + fn))
    fpr.append(fp / (fp + tn))

# Plot ROC curve
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.show()