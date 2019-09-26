import sys, os
from tensorflow import keras

sys.path.append(os.path.join(os.path.realpath(__file__), os.pardir, os.pardir))

IMG_SIZE = 150
NBR_Classes = 16


class CNNForClassif:
    def __init__(self, trainData, trainLabels, validData, validLabels):
        self._trainData = trainData
        self._trainLabels = trainLabels
        self._validData = validData
        self._validLabels = validLabels
        self._model = None

    def GetModel(self):
        return self._model
    def SetModel(self, model):
        self._model = model

    def Static(self, optimizer, lossFunction):
        assert self._model is None
        # Nnet creation
        self._model = keras.models.Sequential()
        self._model.add(keras.layers.Conv2D(32, (3, 3), input_shape=(IMG_SIZE, IMG_SIZE, 3)))
        self._model.add(keras.layers.Activation('relu'))
        self._model.add(keras.layers.Conv2D(32, (3, 3)))
        self._model.add(keras.layers.Activation('relu'))
        self._model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        self._model.add(keras.layers.Dropout(0.25))

        self._model.add(keras.layers.Flatten())  # this converts our 3D feature maps to 1D feature vectors
        self._model.add(keras.layers.Dense(128))
        self._model.add(keras.layers.Activation('relu'))
        self._model.add(keras.layers.Dropout(0.5))
        self._model.add(keras.layers.Dense(16))
        self._model.add(keras.layers.Activation('sigmoid'))

        # Options for training
        self._model.compile(optimizer=optimizer,
                      loss=lossFunction,
                      metrics=['accuracy'])


    def Train(self, nEpochs, jsonPath, h5Path):
        print('beginning training')
        self._model.fit(self._trainData, self._trainLabels,
                        validation_data=(self._validData, self._validLabels),
                        epochs = nEpochs)

        # serialize model to JSON
        modelJson = self._model.to_json()
        with open(jsonPath, "w") as jsonFile:
            jsonFile.write(modelJson)
        # serialize weights to HDF5
        self._model.save_weights(h5Path)

        print('training finished')

def GetCnnFromFiles(jsonFile, h5File):
    # load json and create model
    with open(jsonFile, 'r') as json:
        loaded_model_json = json.read()
    loaded_model = keras.models.model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(h5File)

    return loaded_model








