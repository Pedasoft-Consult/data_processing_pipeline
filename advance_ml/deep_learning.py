# Deep learning
# Import necessary libraries
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense

# Generate dummy data
X = np.random.rand(100, 20)  # 100 samples with 20 features each
y = np.random.randint(0, 10, 100)  # 100 samples with labels (0 to 9)

# One-hot encode the labels
from keras.utils import to_categorical
y = to_categorical(y, num_classes=10)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a neural network
model = Sequential()
model.add(Dense(128, activation='relu', input_dim=X_train.shape[1]))  # Input dimension is the number of features
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))  # Output layer with 10 classes

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
