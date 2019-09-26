import sys, os

sys.path.append(os.path.join(os.path.realpath(__file__), os.pardir, os.pardir))
from Utilities.CreateTrainAndTestData import CreateTrainAndTest

if __name__ == '__main__':
    from optparse import OptionParser

    parser = OptionParser()
    parser.add_option("-i", "--inputJson", dest="inputJson",
                      help="annotated data in json")
    parser.add_option("-t", "--trainFile", dest="trainFile",
                      help="training db file path")
    parser.add_option("-v", "--validFile", dest="validFile",
                      help="validation db file path")
    parser.add_option("-e", "--testFile", dest="testFile",
                      help="test db file path")
    parser.add_option("-s", "--snippetsDir", dest="snippetsDir",
                      help="directory containing the snippets")

    (options, args) = parser.parse_args()
    json = options.inputJson
    trainFile = options.trainFile
    validFile = options.validFile
    testFile = options.testFile
    snippetsDir = options.snippetsDir

    # Create test, train and valid databases for classification
    trainRate = 0.9
    testNbrImgs = 20 # nbr of images per class
    CreateTrainAndTest(json,
                       trainRate, testNbrImgs,
                       trainFile, validFile, testFile,
                       snippetsDir)