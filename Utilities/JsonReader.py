import sys
import cv2
import os
import json
from urllib.request import urlretrieve

sys.path.append(os.path.join(os.path.realpath(__file__), os.pardir))
from Bbox import Bbox

def GetJsonData(jsonFilePath):
    with open(jsonFilePath) as jsonFile:
        imgList = jsonFile.read().split('\n}\n') # separator is lost here
        # Adding the separator in the end of each image data
        imgListCorrected = [imgData + '\n}\n' for imgData in imgList
                   if imgList.index(imgData) != len(imgList) - 1]
        imgListCorrected.append(imgList[len(imgList)-1])
        jsonData = [json.loads(imgData) for imgData in imgListCorrected]

    return jsonData

def GetProductImages(jsonFilePath, imgDir, productImgDir):
    jsonData = GetJsonData(jsonFilePath)
    for imgData in jsonData:
        imgID = imgData['_id']['$oid']
        print('processing %s'%imgID)
        pathToWriteImg = os.path.join(imgDir, imgID) + '.jpg'
        # Dump the full images
        urlretrieve(imgData['url'], pathToWriteImg)
        # Crop the Kukident snippets
        for index, bbox in enumerate(imgData['bounding_boxes']):
            top = int(bbox['ltc']['x']['$numberInt'])
            bottom = int(bbox['rbc']['x']['$numberInt'])
            left = int(bbox['ltc']['y']['$numberInt'])
            right = int(bbox['rbc']['y']['$numberInt'])

            fullImg = cv2.imread(pathToWriteImg)
            croppedImage = fullImg[left:right, top:bottom]

            croppedImgPath = os.path.join(productImgDir, imgID + '_%d.jpg'%(index + 1))
            cv2.imwrite(croppedImgPath, croppedImage)

def GetLabelledDataInJson(jsFile, imgDir):
    dataDict = dict()
    jsonData = GetJsonData(jsFile)
    for imgData in jsonData:
        imgID = imgData['_id']['$oid']
        genericPathToImg = os.path.join(imgDir, imgID) + '.jpg'
        for index, bbox in enumerate(imgData['bounding_boxes']):
            specificImgPath = os.path.join(genericPathToImg + '_%d.jpg'%(index + 1))
            label = bbox['label']
            dataDict[specificImgPath] = label
    return dataDict

def GetImgAndBboxFromJson(jsonFilePath):
    dataDict = dict() # dataDict[imgID] = [bBox1, bBox2, ...]

    jsonData = GetJsonData(jsonFilePath)
    for imgData in jsonData:
        imgID = imgData['_id']['$oid']
        dataDict[imgID] = list()
        for bbox in imgData['bounding_boxes']:
            top = int(bbox['ltc']['x']['$numberInt'])
            bottom = int(bbox['rbc']['x']['$numberInt'])
            left = int(bbox['ltc']['y']['$numberInt'])
            right = int(bbox['rbc']['y']['$numberInt'])
            label = str(bbox['label'])
            dataDict[imgID].append(Bbox(left, right, top, bottom, label))
    return dataDict

def GetAllBboxesToFind(json):
    totalbBoxCounter = dict()
    bBoxDict = GetImgAndBboxFromJson(json)
    allBboxes = bBoxDict.values()

    for imgBboxes in allBboxes:
        for bbox in imgBboxes:
            if bbox._label in totalbBoxCounter.keys():
                totalbBoxCounter[bbox._label] += 1
            else:
                totalbBoxCounter[bbox._label] = 1
    return totalbBoxCounter



if __name__ == '__main__':
    import sys
    import time
    begin = time.time()
    # jsonFilePath = sys.argv[1]
    # imgDir = sys.argv[2]
    # productImgDir = sys.argv[3]
    # GetProductImages(jsonFilePath, imgDir, productImgDir)

    # test on GetTrainDataInJson()
    jsonFile = sys.argv[1]
    imgDir = sys.argv[2]
    GetTrainDataInJson(jsonFile, imgDir)



    end = time.time()
    print('\n\njob\'s done !\n It took {}'.format(end - begin))