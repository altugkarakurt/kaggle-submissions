import numpy as np
import pandas as pd

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout,Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation

"""-------------------- LOADING DATA --------------------"""
test_data = pd.read_csv("../input/mnist_test.csv")
train_data = pd.read_csv("../input/mnist_train.csv")

test_features = np.array(test_data)
train_labels = np.array(train_data["label"])
train_labels = np_utils.to_categorical(train_labels)
train_features = np.array(train_data.drop(columns=["label"]))
print("Data loading done.")

"""-------------------- PREPROCESSING --------------------"""
# scale the pixel values to be in [0, 1]
train_features = train_features / 255.0

# Reshape the 1-D vector data to recover the square images
train_features = train_features.reshape(train_features.shape[0], 28, 28 , 1).astype('float32')
print("Training preprocessing done.")

"""-------------------- CNN TRAINING --------------------"""
model = Sequential()

# First convolutional layer
model.add(Conv2D(40, kernel_size=5, padding="same",input_shape=(28, 28, 1), activation = 'relu'))
model.add(Conv2D(50, kernel_size=5, padding="valid", activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.2))

# Second Pooling layer
model.add(Conv2D(70, kernel_size=3, padding="same", activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(100, kernel_size=3, padding="valid", activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Flatten layer
model.add(Flatten())
model.add(Dense(100))
model.add(Activation("relu"))
model.add(Dropout(0.2))

model.add(Dense(10))
model.add(Activation("softmax"))
print("CNN configuration done.")

# Final configuration and training
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_features, train_labels, epochs= 32 , batch_size=200, validation_split = 0.2)
print("Training done.")

"""-------------------- TESTING --------------------"""
# scale the pixel values to be in [0, 1]
test_features = test_features / 255.0

# Reshape the 1-D vector data to recover the square images
test_features = test_features.reshape(test_features.shape[0], 28, 28 , 1).astype('float32')
print("Testing preprocessing done.")

test_labels = model.predict(test_features)
print("Testing done.")
test_labels = np.argmax(test_labels,axis = 1)
test_ids = np.arange(1, len(test_features)+1)
df = pd.DataFrame(data = {"ImageId":test_ids, "Label":test_labels})
df.to_csv("cnn.csv", index=False)