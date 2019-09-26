

class Bbox:
    def __init__(self, left, right, top, bottom, label):
        self._left = left
        self._right = right
        self._top = top
        self._bottom = bottom
        self._label = label

    def SetLabel(self, label):
        self._label = label
    def GetLabel(self):
        return self._label
