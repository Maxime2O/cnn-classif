import os, sys

# Project imports
sys.path.append(os.path.join(os.path.realpath(__file__), os.pardir, os.pardir))
from Utilities.CreateTrainAndTestData import FormatDataForTensorflow
from Classification.classification import CNNForClassif, GetCnnFromFiles, IMG_SIZE

LOSS_FUNCTION = 'categorical_crossentropy'

if __name__ == '__main__':
    from optparse import OptionParser
    parser = OptionParser()

    # Train
    parser.add_option("-d", "--dir", dest="dumpDir",
                      help="dir to write the models")
    parser.add_option("-t", "--train", dest="trainDb",
                      help="training database")
    parser.add_option("-v", "--valid", dest="validDb",
                      help="validation database")

    (options, args) = parser.parse_args()
    trainDb = options.trainDb
    validDb = options.validDb
    dumpModelDir = options.dumpDir

    if trainDb is None:
        raise ValueError('A training db is mandatory')
    if validDb is None:
        raise ValueError('A validation db is mandatory')

    trainData, trainLabels = FormatDataForTensorflow(trainDb, IMG_SIZE)
    validData, validLabels = FormatDataForTensorflow(validDb, IMG_SIZE)

    print('----- Data preparation ok')

    # optimizers = ['adam', 'sgd', 'rmsprop']
    optimizers = ['adam']
    for optimizer in optimizers:
        cnn = CNNForClassif(trainData, trainLabels, validData, validLabels)
        print('----- Cnn created')
        cnn.Static(optimizer = optimizer, lossFunction = LOSS_FUNCTION)
        print('----- Static ok')

        # save the initial model
        modelJson = cnn._model.to_json()
        with open(os.path.join(dumpModelDir,'cnn_init_%s.json'%optimizer), "w") as jsonFile:
            jsonFile.write(modelJson)
        cnn._model.save_weights(os.path.join(dumpModelDir,'cnn_init_%s.h5'%optimizer))

        # nEpochs = [10, 20, 30, 40, 50]
        nEpochs = [1, 2]
        for index, nEpoch in enumerate(nEpochs):
            # load initial model
            if index != 0:
                initModel = GetCnnFromFiles(os.path.join(dumpModelDir,'cnn_init_%s.json'%optimizer),
                                      os.path.join(dumpModelDir, 'cnn_init_%s.h5'%optimizer))
                cnn.SetModel(initModel)
                cnn._model.compile(optimizer = optimizer,
                          loss = LOSS_FUNCTION,
                          metrics = ['accuracy'])
            cnn.Train(nEpoch,
                      os.path.join(dumpModelDir,'cnn_{}_epochs_{}.json'.format(nEpoch, optimizer)),
                      os.path.join(dumpModelDir,'cnn_{}_epochs_{}.h5'.format(nEpoch, optimizer)))

    print('job\'s done!')