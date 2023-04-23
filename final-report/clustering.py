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
df = df.drop(columns=['name', 'preference', 'key', 'mode'])

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

# Set the number of clusters (k)
k = 3

# Initialize the KMeans object
kmeans = KMeans(n_clusters=k)

# Fit the data to the KMeans object
kmeans.fit(X)

# Get the cluster labels for each data point
labels = kmeans.labels_

# Add the cluster labels to the original data as a new column
df['cluster'] = labels

# Print the number of data points in each cluster
print(df['cluster'].value_counts())

# Group the data by cluster
clusters = df.groupby('cluster')

# Iterate over the clusters and print the details
for cluster, cluster_data in clusters:
    print('Cluster {}:'.format(cluster))
    print('Number of data points: {}'.format(len(cluster_data)))
    print('Mean values of features:')
    print(cluster_data.mean())
    print('\n')

# Perform PCA to project the data onto two principal components
pca = PCA(n_components=2)
pca_data = pca.fit_transform(X)

# Plot the data points, colored by cluster
plt.scatter(pca_data[:,0], pca_data[:,1], c=df['cluster'])

# Define a dictionary that maps cluster labels to colors
colors = {0: 'yellow', 1: 'purple', 2: 'teal'}

# Define a value for the marker size
marker_size = 25
#marker_alpha = 1

# Plot the data points, colored by cluster and with larger markers
plt.scatter(pca_data[:,0], pca_data[:,1], c=df['cluster'].map(colors), s=marker_size)

# Create a legend that shows the cluster labels and colors
legend_elements = [plt.Line2D([0], [0], marker='o', color=color, label='Cluster {}'.format(cluster))
                   for cluster, color in colors.items()]
plt.legend(handles=legend_elements)

# Add axis labels and a title to the plot
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('K-Means Clustering Results')

# Get the principal axes in feature space
feature_vectors = pca.components_.T

# Set scaling factor for feature vectors
scale_factor = 4

# Plot the feature vectors
for i, feature_vector in enumerate(feature_vectors):
    plt.arrow(0, 0, feature_vector[0]*scale_factor, feature_vector[1]*scale_factor, 
              color='r', alpha=0.5, linewidth=2*scale_factor, 
              head_width=0.1*scale_factor, head_length=0.1*scale_factor)
    plt.text(feature_vector[0]*scale_factor*1.15, feature_vector[1]*scale_factor*1.15, 
             df.columns[:-1][i], color='r', ha='center', va='center')

plt.savefig('kmeanscluster.png')
# Show the plot
plt.show()
