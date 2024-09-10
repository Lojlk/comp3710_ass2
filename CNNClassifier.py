import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt


# CNN classifier
# Load the LFW dataset
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
X = lfw_people.images
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = len(target_names)

# Check the value range of X
print("X_min:", X.min(), "X_max:", X.max())

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Add an extra dimension for the channel (required by Conv2D)
X_train = X_train[:, :, :, np.newaxis]  # shape will be (n_samples, height, width, 1)
X_test = X_test[:, :, :, np.newaxis]

# Network parameters
h, w, c = X_train.shape[1:]  # height, width, channels (c will be 1 for grayscale)
batch_size = 64
epochs = 20

# Build the CNN model
inputs = Input(shape=(h, w, c))  # Image input (resized LFW images)
net1 = Conv2D(32, (3, 3), padding="same", activation="relu")(inputs)
pool1 = MaxPooling2D(pool_size=(2, 2))(net1)
net2 = Conv2D(32, (3, 3), padding="same", activation="relu")(pool1)
pool2 = MaxPooling2D(pool_size=(2, 2))(net2)
flat = Flatten()(pool2)
net3 = Dense(128, activation="relu")(flat)
output = Dense(n_classes, activation="softmax")(net3)

# Create the model
model = Model(inputs, output)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, shuffle=True, validation_data=(X_test, y_test))

# Evaluate the model on the test set and print the accuracy
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Make predictions on the test set
predictions = model.predict(X_test)

# Convert predictions to class labels (argmax of softmax output)
predicted_labels = np.argmax(predictions, axis=1)

# Print the classification report
print("Classification Report:")
print(classification_report(y_test, predicted_labels, target_names=target_names))
