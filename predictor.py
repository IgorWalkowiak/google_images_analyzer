import numpy as np
from tensorflow.keras import models
import cv2 as cv

classes=['forest', 'ocean', 'sun']
label = [x for x in range(len(classes))]
indexToClass = dict(zip(label,classes))

picName = 'image.jpg' 
model = models.load_model('first.model')

src = cv.imread(picName, cv.COLOR_BGR2RGB)
dsize = (64, 64)
output = cv.resize(src, dsize)
data = np.asarray([output]) / 255
prediction = model.predict(data)
index = np.argmax(prediction)
print(prediction[0])
print(indexToClass[index])