import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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

# Perform PCA with 2 components
pca = PCA(n_components=2)
principal_components = pca.fit_transform(X)

# Create a scatter plot of the feature vectors
plt.scatter(principal_components[:, 0], principal_components[:, 1])
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA of the data")

# Get the principal axes in feature space
feature_vectors = pca.components_.T

# Set scaling factor for feature vectors
scale_factor = 3

# Plot the feature vectors
for i, feature_vector in enumerate(feature_vectors):
    plt.arrow(0, 0, feature_vector[0]*scale_factor, feature_vector[1]*scale_factor, 
              color='r', alpha=0.5, linewidth=2*scale_factor, 
              head_width=0.1*scale_factor, head_length=0.1*scale_factor)
    plt.text(feature_vector[0]*scale_factor*1.15, feature_vector[1]*scale_factor*1.15, 
             df.columns[:-1][i], color='r', ha='center', va='center')

plt.savefig('pca.png')
plt.show()