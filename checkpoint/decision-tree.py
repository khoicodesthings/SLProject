import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Node:
    def __init__(self, data, target):
        self.left = None
        self.right = None
        self.data = data
        self.target = target
        self.feature = None
        self.threshold = None
        self.leaf = False
        self.prediction = None

class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, data, target):
        self.root = self.build_tree(data, target, depth=0)

    def build_tree(self, data, target, depth):
        num_samples, num_features = data.shape
        num_pos = (target == 1).sum()
        num_neg = (target == 0).sum()

        # base cases
        if depth == self.max_depth or num_samples <= 1 or num_pos == 0 or num_neg == 0:
            leaf = Node(data, target)
            leaf.leaf = True
            leaf.prediction = 1 if num_pos > num_neg else 0
            return leaf

        # find best feature and threshold to split on
        best_feature = None
        best_threshold = None
        best_gini = 1.0

        for feature in range(num_features):
            thresholds = np.unique(data[:, feature])
            for threshold in thresholds:
                left_indices = data[:, feature] <= threshold
                right_indices = data[:, feature] > threshold

                if left_indices.sum() == 0 or right_indices.sum() == 0:
                    continue

                left_target = target[left_indices]
                right_target = target[right_indices]

                gini_left = 1.0 - ((left_target == 1).sum() / left_target.shape[0])**2 - ((left_target == 0).sum() / left_target.shape[0])**2
                gini_right = 1.0 - ((right_target == 1).sum() / right_target.shape[0])**2 - ((right_target == 0).sum() / right_target.shape[0])**2

                weighted_gini = (left_target.shape[0] / num_samples) * gini_left + (right_target.shape[0] / num_samples) * gini_right

                if weighted_gini < best_gini:
                    best_feature = feature
                    best_threshold = threshold
                    best_gini = weighted_gini

        # split on best feature and threshold
        left_indices = data[:, best_feature] <= best_threshold
        right_indices = data[:, best_feature] > best_threshold

        # grow subtrees
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

# Calculate precision and recall
def true_positives(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    return tp

def false_positives(y_true, y_pred):
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return fp

# Load dataset from final.csv file
df = pd.read_csv('final.csv')

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

# Train the decision tree
tree = DecisionTree(max_depth=3)
tree.fit(X_train, y_train)

# Make predictions on the validation set
y_pred = tree.predict(X_val)

# Evaluate the performance of the model on the validation set
accuracy = np.mean(y_pred == y_val)
print("Accuracy:", accuracy)

tp = true_positives(y_val, y_pred)
fp = false_positives(y_val, y_pred)

precision = tp / (tp + fp)
recall = tp / (tp + np.sum(y_val == 1))

print('Precision:', precision)
print('Recall:', recall)

# Calculate TPR and FPR at different thresholds
thresholds = np.linspace(0, 1, 100)
tpr = []
fpr = []
for threshold in thresholds:
    y_pred_threshold = (y_pred >= threshold).astype(int)
    tp = true_positives(y_val, y_pred_threshold)
    fp = false_positives(y_val, y_pred_threshold)
    tn = np.sum((y_val == 0) & (y_pred_threshold == 0))
    fn = np.sum((y_val == 1) & (y_pred_threshold == 0))
    tpr.append(tp / (tp + fn))
    fpr.append(fp / (fp + tn))

# Plot ROC
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Decision Trees ROC Curve')
plt.show()







