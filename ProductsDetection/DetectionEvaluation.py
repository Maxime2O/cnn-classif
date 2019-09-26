

def CalculateIoU(bb1, bb2):
    # coordinates of the intersection rectangle
    x1 = max(bb1._left, bb2._left)
    y1 = max(bb1._top, bb2._top)
    x2 = min(bb1._right, bb2._right)
    y2 = min(bb1._bottom, bb2._bottom)

    # compute the area of intersection rectangle
    interArea = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

    # compute the area of both the prediction and ground-truth rectangles
    bb1Area = (bb1._right - bb1._left + 1) * (bb1._bottom - bb1._top + 1)
    bb2Area = (bb2._right - bb2._left + 1) * (bb2._bottom - bb1._top + 1)

    # compute the intersection over union
    iou = interArea / float(bb1Area + bb2Area - interArea)

    return iou

def AreMatching(bb1, bb2):
    # Criterion: intersection over union > 0.5
    iou = CalculateIoU(bb1, bb2)
    if iou > 0.5:
        return True
    else:
        return False


def EvaluationOnImage(BboxFound, BboxOnImage):
    for bbGT in BboxOnImage:
        for bb in BboxFound:
            if AreMatching(bb, bbGT):
                return bb
    return None