
import os, sys

sys.path.append(os.path.join(os.path.realpath(__file__), os.pardir, os.pardir))
from Utilities.JsonReader import GetProductImages
from Utilities.CreateTrainAndTestData import CreateTrainAndTest

if __name__ == '__main__':
    from optparse import OptionParser

    parser = OptionParser()
    parser.add_option("-i", "--inputJson", dest="inputJson",
                      help="annotated data in json")
    parser.add_option("-d", "--imgDir", dest="imgDir",
                      help="directory to dump the full images")
    parser.add_option("-s", "--snippetDir", dest="snippetDir",
                      help="directory to dump the snippets")

    (options, args) = parser.parse_args()
    json = options.inputJson
    imgDir = options.imgDir
    dirToDumpSnippets = options.snippetDir

    # Dump the images and the snippets
    GetProductImages(json, imgDir, dirToDumpSnippets)