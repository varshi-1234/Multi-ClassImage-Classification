# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pickle
import random
import os
from keras.datasets import cifar10
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical
from sklearn.metrics import classification_report
from skimage.transform import resize

# 1. Data Processing
# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Visualize a random image
random_index = random.randint(0, len(x_train) - 1)
plt.imshow(x_train[random_index])
plt.title(f'The label of this category is: {y_train[random_index]}')
plt.show()

# One-hot encode the labels
y_train_one_hot = to_categorical(y_train, 10)
y_test_one_hot = to_categorical(y_test, 10)

# Scale the pixel values
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Create Data directory if it doesn't exist
if not os.path.exists("Data"):
    os.makedirs("Data")

# Save the test data
pickle.dump(x_test, open("Data/x_test.dat", "wb"))
pickle.dump(y_test_one_hot, open("Data/y_test.dat", "wb"))

# 2. Building and Training the CNN
model = Sequential()

# Add layers
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
hist = model.fit(x_train, y_train_one_hot, batch_size=32, epochs=20, validation_split=0.2)

# 3. Evaluate the model
# Plot training & validation accuracy values
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Test the model
test_loss, test_accuracy = model.evaluate(x_test, y_test_one_hot)
print(f'The accuracy of the model on the test set is: {test_accuracy * 100:.2f}%')

# Save the trained model
model.save('model_name.h5')  # Change 'model_name.h5' to your desired file name

# 4. Prediction and Evaluation on Test Set
# Load the trained model
model = load_model('model_name.h5')  # Change to your model filename

# Load the test data
x_test = pickle.load(open("Data/x_test.dat", "rb"))
y_test = pickle.load(open("Data/y_test.dat", "rb"))

# Get the true labels
y_test_label = np.argmax(y_test, axis=1)

# Predict on the test set
predicted_classes = model.predict(x_test)
predicted_classes_label = np.argmax(predicted_classes, axis=1)

# Compare correct and incorrect answers
correct = np.where(predicted_classes_label == y_test_label)[0]
print("Found", len(correct), "correct classes")
incorrect = np.where(predicted_classes_label != y_test_label)[0]
print("Found", len(incorrect), "incorrect classes")

# Visualizing a random incorrect prediction
if len(incorrect) > 0:
    random_incorrect = random.choice(incorrect)
    plt.imshow(x_test[random_incorrect])
    plt.title(f'Predicted: {predicted_classes_label[random_incorrect]}, Actual: {y_test_label[random_incorrect]}')
    plt.axis('off')
    plt.show()

# Classification report
target_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
print(classification_report(y_test_label, predicted_classes_label, target_names=target_names))
