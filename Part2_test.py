
import os, sys
import numpy as np
import tensorflow as tf
from tensorflow import keras

sys.path.append(os.path.join(os.path.realpath(__file__), os.pardir, os.pardir))
from Classification.classification import GetCnnFromFiles, IMG_SIZE
from Utilities.CreateTrainAndTestData import FormatDataForTensorflow

def TestCnn(jsonFile, weightFile):
    print('getting cnn from %s'%jsonFile)
    cnnModel = GetCnnFromFiles(jsonFile, weightFile)
    print('cnnModel loaded')

    optimizer = jsonFile.strip('.json').split('_')[-1]
    cnnModel.compile(loss='categorical_crossentropy', optimizer=optimizer,
                     metrics=['accuracy'])

    testData, testLabels = FormatDataForTensorflow(testDb, IMG_SIZE)


    score = cnnModel.evaluate(testData, testLabels)
    print("%s: %.2f%%" % (cnnModel.metrics_names[1], score[1] * 100))
    # Get the confusion matrix
    labels_pred = cnnModel.predict_classes(testData)

    labels_pred = np.array(labels_pred, dtype=np.float)
    labels_pred = keras.utils.to_categorical(labels_pred, 16)

    con_mat = tf.confusion_matrix(labels=tf.argmax(testLabels, 1),
                                       predictions=tf.argmax(labels_pred, 1))
    sess = tf.Session()

    return con_mat.eval(session = sess)

if __name__ == '__main__':
    from optparse import OptionParser

    parser = OptionParser()
    ## Test
    parser.add_option("-i", "--inputDir", dest="modelDir",
                      help="directory containing the model files")
    parser.add_option("-o", "--outputFile", dest="resultFile",
                      help="file to dump the results")
    parser.add_option("-t", "--test", dest="test",
                      help="test db")

    (options, args) = parser.parse_args()
    modelDir = options.modelDir
    resultFile = options.resultFile
    testDb = options.test

    with open(resultFile, 'w') as oFile:
        for file in os.listdir(modelDir):
            if '.json' in file:
                cnnId = file.strip('.json')
                weightFile = cnnId + '.h5'

                # build complete paths
                jsonFilePath = os.path.join(modelDir, file)
                h5FilePath = os.path.join(modelDir, weightFile)
                conMat = TestCnn(jsonFilePath, h5FilePath)

                oFile.write('{} : {}\n'.format(cnnId, conMat))



