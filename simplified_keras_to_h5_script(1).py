
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.utils import to_categorical

# Generate synthetic dataset
num_samples = 1000
img_shape = (48, 48, 1)
X = np.random.randint(0, 256, size=(num_samples, *img_shape)).astype('float32')
y = np.random.randint(0, 7, size=(num_samples,))
y_onehot = to_categorical(y, num_classes=7)

# Normalize the data
X /= 255.0

# Build a simplified model
model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(7, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X, y_onehot, epochs=45, batch_size=64, validation_split=0.8)

# Save the model as .h5 file
model.save('simplified_emotion_model.h5')
