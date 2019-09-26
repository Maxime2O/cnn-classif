import sys, os
import random
import cv2
import numpy as np
from tensorflow import keras

# project imports
sys.path.append(os.path.join(os.path.realpath(__file__), os.pardir))
from JsonReader import GetLabelledDataInJson
from ClassDict import CLASS_DICT


def CreateTestDbInTxt(data, testFile, nbrImgsPerLabel):
    ## Get test data
    labelList = [imgData[1] for imgData in data]
    testImgs = list()
    testDict = dict() # testDict[label] = [imgs]
    for (img, label) in data:
        if label not in testDict.keys():
            if labelList.count(label) > nbrImgsPerLabel:
                testDict[label] = [img]
                testImgs.append(img)
            else:
                pass
        else:
            if len(testDict[label]) < nbrImgsPerLabel < labelList.count(label):
                        testDict[label].append(img)
                        testImgs.append(img)

    ## Write test data
    with open(testFile, 'w') as test:
        for (label, imgs) in testDict.items():
            for img in imgs:
                test.write('{} {}\n'.format(img, label))
    return testImgs


def CreateTrainDbInTxt(dataInJson,
                         trainRate, trainTxt, validTxt):
    # trainRate : % of the train data
    nbrImgs = len(dataInJson)
    nbrTrain = int(trainRate * nbrImgs)

    with open(trainTxt, 'w') as trainFile:
        with open(validTxt, 'w') as validFile:
            for index, (img, label) in enumerate(dataInJson):
                if index < nbrTrain:
                    trainFile.write('{} {}\n'.format(img, label))
                else:
                    validFile.write('{} {}\n'.format(img, label))


def CreateTrainAndTest(json,
                       trainRate, testNbrImgs,
                       trainFile, validFile, testFile,
                       imgDir):
    # Keep testNbrImgs images per label for test
    labelledDataInJson = GetLabelledDataInJson(json, imgDir)
    imgs = list(labelledDataInJson.keys())
    random.shuffle(imgs)
    dataInJson = [(img, labelledDataInJson[img]) for img in imgs]

    testImgs = CreateTestDbInTxt(dataInJson, testFile,  nbrImgsPerLabel = testNbrImgs)
    dataInJson = [(img, labelledDataInJson[img]) for img in imgs
                  if img not in testImgs]
    CreateTrainDbInTxt(dataInJson, trainRate, trainFile, validFile)

def FormatDataForTensorflow(dataDb, imgSize):
    data = list()
    labels = list()
    with open(dataDb) as db:
        for line in db.readlines():
            imgPath = line.split()[0]
            assert os.path.isfile(imgPath)
            className = ' '.join(line.split()[1:])

            # format the label
            classInt = int(CLASS_DICT[className])

            # Get the image
            img = cv2.imread(imgPath)
            if img is not None:
                img = cv2.resize(img, (imgSize, imgSize))
                data.append(img)
                labels.append(classInt)

    # format for Keras
    data = np.array(data, dtype=np.float).reshape(-1, imgSize, imgSize, 3)
    labels = np.array(labels, dtype=np.float)
    labels = keras.utils.to_categorical(labels, 16)

    return data, labels


