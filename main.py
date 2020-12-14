import numpy as np
from tensorflow.keras import layers, models

import data

classes=['forest', 'ocean', 'sun']
(training_data, training_labels), (test_data, test_labels) = data.getData(classes, 0.001, 20)

model = models.Sequential()
model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(64,64,3) ))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(len(classes), activation='softmax'))

training_labels = np.array(training_labels)
training_data = np.array(training_data)


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(training_data, training_labels, epochs=30, validation_split=0.2)
model.save('first.model')
