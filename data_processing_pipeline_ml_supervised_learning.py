#Complete Pipeline: Data Collection, Cleaning, Transformation, Splitting, and Supervised Learning

# Import necessary libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

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
# Defining feature columns (X) and target variable (y)
if 'target' in data.columns:  # Ensure the target column exists
    X = data.drop('target', axis=1)
    y = data['target']

    # Splitting the dataset into 80% training and 20% testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\nData splitting complete.")
    print(f"Training data shape: {X_train.shape}, Testing data shape: {X_test.shape}")

# Supervised learning
# Create and train a linear regression model
model = LinearRegression()

# Train the model using the training dataset
model.fit(X_train, y_train)

# Make predictions on the test dataset
y_pred = model.predict(X_test)

# Display the first few predictions
print("\nPredictions on test data:")
print(y_pred[:10])

# Save the cleaned and transformed dataset to a new file (optional)
data.to_csv('cleaned_transformed_data.csv', index=False)
print("\nCleaned and transformed dataset saved as 'cleaned_transformed_data.csv'.")
