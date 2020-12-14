import images as imgs
import os
import random
from keras_preprocessing import image

IMAGES_DIR = 'google_photos'


def _doesImagesExists(className, imagesPerClass):
    try:
        imagesFolder = os.listdir(IMAGES_DIR)
        if imagesFolder.count(className):
            if len(os.listdir(IMAGES_DIR+'/'+className)) > int(imagesPerClass*0.8):
                return True
    except Exception as e:
        print(str(e))
    return False


def _shuffleData(data, labels):
    data_labels = zip(data, labels)
    zipped = [*data_labels]
    random.shuffle(zipped)
    data, labels = zip(*zipped)
    data = list(data)
    labels = list(labels)
    return data, labels

def _downloadData(classNames, imagesPerClass):
    images = imgs.Images(IMAGES_DIR, 64, 64, imagesPerClass)
    for className in classNames:
        if not _doesImagesExists(className, imagesPerClass):
            images.downloadImages(className)

def getData(classNames, testRatio, imagesPerClass):
    _downloadData(classNames, imagesPerClass)

    label = [x for x in range(len(classNames))]
    classToLabel = dict(zip(classNames, label))
    outputData = []
    outputLabels = []
    for className in classNames:
        class_dir = IMAGES_DIR + '/' + className
        arr = os.listdir(class_dir)
        classLabel = classToLabel[className]

        count = 0
        for imgDir in arr:
            try:
                if count == imagesPerClass:
                    break
                img = image.load_img(class_dir + '/' + imgDir)
                x = image.img_to_array(img)
                x = x/255
                outputData.append(x)
                outputLabels.append([classLabel])
                count = count + 1
            except Exception as e:
                print(str(e))


    testSize = int(len(outputData) * testRatio)
    outputData, outputLabels = _shuffleData(outputData, outputLabels)
    outputTrainData = outputData[-testSize:]
    outputTrainLabel = outputLabels[-testSize:]
    outputTestData = outputData[:-testSize]
    outputTestLabel = outputLabels[:-testSize]
    return (outputTrainData, outputTrainLabel), (outputTestData, outputTestLabel)