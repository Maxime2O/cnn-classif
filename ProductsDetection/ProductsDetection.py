import sys, os
sys.path.append(os.path.join(os.path.realpath(__file__), os.pardir))
import cv2

from Utilities.Bbox import Bbox

X_CANNY = 30
Y_CANNY = 150
NBR_FEATURES = 50000

# Margins for the extension of points found with ORB
X_MARGIN = 40
Y_MARGIN = 80

def PreprocImg(img):
    # Grayscale and edge detection
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    imgCanny = cv2.Canny(imgGray, X_CANNY, Y_CANNY)

    return imgCanny

def GetFoundObjectBbox(kpt, matches, img, className):
    imgSizeX = img.shape[0]
    imgSizeY = img.shape[1]
    bBoxFound = list()

    for match in matches:
        # Get the coordinates of the matching point
        imgIdx = match.trainIdx # index of the keypoint matching in the image
        (y, x) = kpt[imgIdx].pt

        ## Create a bounding box
        xMarginLow = xMarginHigh = X_MARGIN
        yMarginLow = yMarginHigh = Y_MARGIN
        # Handelling border problems
        if x-X_MARGIN < 0:
            xMarginLow = x
        if x+X_MARGIN > imgSizeX:
            xMarginHigh = imgSizeX - x
        if y-Y_MARGIN < 0:
            yMarginLow = y
        if y+Y_MARGIN > imgSizeY:
            yMarginHigh = imgSizeY - y

        # Create the bounding boxes
        horizontalBbox = Bbox(int(x-xMarginLow), int(x+xMarginHigh),
                              int(y - yMarginLow), int(y+yMarginHigh), className)
        verticalBbox = Bbox(int(x-yMarginLow), int(x+yMarginHigh),
                            int(y - xMarginLow), int(y+xMarginHigh), className)
        bBoxFound.append(horizontalBbox)
        bBoxFound.append(verticalBbox)

    return bBoxFound

def ORBReference(refImg):
    # Get the ORB keypoints on reference images
    orbRef = cv2.ORB_create()
    kpRef, desRef = orbRef.detectAndCompute(refImg, None)
    return kpRef, desRef

def ORBDetection(img, nFeatures):
    # ORB method
    orb = cv2.ORB_create(nfeatures = nFeatures)
    kpFound, desFound = orb.detectAndCompute(img, None)
    return kpFound, desFound


def ORBMatching(desRef, desFound):
    # Brute-Force Matching with SIFT Descriptors and Ratio Test
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desRef,desFound, k=2)
    
    # Apply ratio test
    goodMatches = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            goodMatches.append(m)
            
    return goodMatches


