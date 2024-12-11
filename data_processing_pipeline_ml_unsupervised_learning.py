# Complete Pipeline: Data Collection, Cleaning, Transformation, Splitting, and Unsupervised Learning


# Import necessary libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

# Data collection
# Reading the dataset
data = pd.read_csv('data.csv')

# Displaying the first few rows of the dataset
print("Initial data preview:")
print(data.head())

# Data cleaning
# Removing duplicates
data = data.drop_duplicates()

# Handling missing values by replacing them with the mean
data.fillna(data.mean(), inplace=True)

# Verify no missing values
print("\nMissing values after cleaning:")
print(data.isnull().sum())

# Display the cleaned data preview
print("\nCleaned data preview:")
print(data.head())

# Data transformation
# Encoding categorical columns
label_encoder = LabelEncoder()
if 'category' in data.columns:  # Ensure the column exists before applying transformations
    data['category_encoded'] = label_encoder.fit_transform(data['category'])

# Feature creation (e.g., creating a new feature based on existing ones)
if 'feature1' in data.columns and 'feature2' in data.columns:  # Check if columns exist
    data['new_feature'] = data['feature1'] * data['feature2']

# Display the transformed dataset
print("\nTransformed data preview:")
print(data.head())

# Data splitting
# Defining feature columns (X) and target variable (y) if 'target' exists
if 'target' in data.columns:  # Ensure the target column exists
    X = data.drop('target', axis=1)
    y = data['target']

    # Splitting the dataset into 80% training and 20% testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\nData splitting complete.")
    print(f"Training data shape: {X_train.shape}, Testing data shape: {X_test.shape}")
else:
    # If 'target' does not exist, use all columns for clustering
    X = data

# Unsupervised learning
# Create a KMeans model for clustering
kmeans = KMeans(n_clusters=3, random_state=42)  # Clustering into 3 groups

# Train the model on the feature set
kmeans.fit(X)

# Predict the clusters for each data point
clusters = kmeans.predict(X)

# Add the cluster assignments as a new column in the dataset
data['cluster'] = clusters

# Display the dataset with cluster assignments
print("\nDataset with cluster assignments:")
print(data.head())

# Save the cleaned, transformed, and clustered dataset to a new file (optional)
data.to_csv('final_data_with_clusters.csv', index=False)
print("\nFinal dataset saved as 'final_data_with_clusters.csv'.")
