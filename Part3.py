import os, sys
import cv2

sys.path.append(os.path.join(os.path.realpath(__file__), os.pardir))


# project imports
from ProductsDetection.ProductsDetection import PreprocImg, ORBReference, ORBDetection, ORBMatching, NBR_FEATURES, GetFoundObjectBbox
from ProductsDetection.DetectionEvaluation import EvaluationOnImage
from Utilities.JsonReader import GetImgAndBboxFromJson, GetAllBboxesToFind
from Utilities.ClassDict import CLASS_DICT


if __name__ == '__main__':
    from optparse import OptionParser
    parser = OptionParser(usage="usage: ProductsDetections -j json -d refDir -i imgDir -r resultFiles")

    # Train
    parser.add_option("-j", "--json", dest="json",
                      help="json annotated file")
    parser.add_option("-d", "--refDir", dest="refDir",
                      help="directory contaning reference image for each class")
    parser.add_option("-i", "--imgDir", dest="imgDir",
                      help="directory contaning images to analyze")
    parser.add_option("-r", "--resFile", dest="resFile",
                      help="file containing the results of the evaluation")

    (options, args) = parser.parse_args()
    dataJson = options.json
    refDir = options.refDir
    imgDir = options.imgDir
    assert refDir is not None
    assert dataJson is not None
    assert imgDir is not None

    bboxFoundCounter = dict()
    bboxOkCounter = dict()

    # Get the reference keypoints models
    desRefs = list()
    kptRefs = list()
    for imgName in os.listdir(refDir):
        className = imgName.strip('.jpg')
        className = ' '.join(className.split('_'))

        imgPath = os.path.join(refDir, imgName)
        img = cv2.imread(imgPath)
        img = PreprocImg(img)

        kpRef, desRef = ORBReference(img)
        kptRefs.append(kpRef)
        desRefs.append((desRef, className))

    # Process the dataset
    bBoxDict = GetImgAndBboxFromJson(dataJson)
    for imgID in bBoxDict.keys():
        print('processing %s'%imgID)
        img = cv2.imread(os.path.join(imgDir, imgID) + '.jpg')
        img = PreprocImg(img)
        allBboxes = bBoxDict[imgID]

        # Detection
        kpFound, desFound = ORBDetection(img, NBR_FEATURES)
        for desRef, className in desRefs:
            matches = ORBMatching(desRef, desFound)
            if matches:
                bBoxes = GetFoundObjectBbox(kpFound, matches, img, className)
                if className in bboxFoundCounter.keys():
                    bboxFoundCounter[className] += len(bBoxes)/2 # because vertical and horizontal boxes
                else:
                    bboxFoundCounter[className] = len(bBoxes)/2 # because vertical and horizontal boxes

                # Evaluation
                bboxOk = EvaluationOnImage(bBoxes, allBboxes)
                if bboxOk:
                    if className in bboxOkCounter.keys():
                        bboxOkCounter[className] += 1
                    else:
                        bboxOkCounter[className] = 1

    totalbBoxCounter = GetAllBboxesToFind(dataJson)
    for className in CLASS_DICT.keys():
        if className not in totalbBoxCounter.keys():
            print('class %s not in test db'%className)
        elif className not in bboxFoundCounter.keys():
            print('class %s was not detected' % className)
        else:
            if className not in bboxOkCounter.keys():
                bboxOkCounter[className] = 0
            print('Precision {} : {}'.format(className, bboxOkCounter[className]/bboxFoundCounter[className]))
            print('Recall {} : {}'.format(className, bboxOkCounter[className]/totalbBoxCounter[className]))
