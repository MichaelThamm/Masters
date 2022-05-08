from LIM.Show import *
import json
import os
from platypus import *

PROJECT_PATH = os.path.abspath(os.path.join(__file__, "../.."))
OUTPUT_PATH = os.path.join(PROJECT_PATH, 'Output')
DATA_PATH = os.path.join(OUTPUT_PATH, 'StoredSolutionData.json')

'''
https://platypus.readthedocs.io/en/latest/getting-started.html
'''


class EncoderDecoder(object):
    def __init__(self, model):

        self.encoded = False
        self.rawModel = model
        self.rebuiltModel = None

        self.removedAttributesList = ['matrixA', 'matrixB', 'matrixX']

        # These members are the dictionary results of the json serialization format required in json.dump
        self.encodedAttributes = {}

        self.typeList = []
        self.unacceptedTypeList = [complex, np.complex128]

        # This dict contains the attributes that contain unaccepted objects as values
        self.unacceptedJsonAttributes = {'matrix': Node, 'errorDict': (TransformedDict, Error), 'hmUnknownsList': Region}
        self.unacceptedJsonObjects = []
        self.__setFlattenJsonObjects()

    def __setFlattenJsonObjects(self):
        for value in self.unacceptedJsonAttributes.values():
            try:
                for nestedVal in value:
                    self.unacceptedJsonObjects.append(nestedVal)
            except TypeError:
                self.unacceptedJsonObjects.append(value)

    def __convertComplexInList(self, inList):
        # 'plex_Signature' is used to identify if a list is a destructed complex number or not
        return list(map(lambda x: ['plex_Signature', x.real, x.imag], inList))

    def __objectToDict(self, attribute, originalValue):
        if attribute == 'errorDict':
            errorDict = originalValue.__dict__['store']
            returnVal = {key: error.__dict__ for key, error in errorDict.items()}
        elif attribute == 'matrix':
            listedMatrix = originalValue.tolist()
            # Case 2D list
            for idxRow, row in enumerate(listedMatrix):
                for idxCol, col in enumerate(row):
                    listedMatrix[idxRow][idxCol] = {key: value if type(value) not in self.unacceptedTypeList else ['plex_Signature', value.real, value.imag] for key, value in row[idxCol].__dict__.items()}
            returnVal = listedMatrix
        elif attribute == 'hmUnknownsList':
            returnVal = {}
            tempDict = {}
            for regionKey, regionVal in originalValue.items():
                for dictKey, dictVal in regionVal.__dict__.items():
                    if dictKey not in ['an', 'bn']:
                        tempDict[dictKey] = dictVal
                    else:
                        tempDict[dictKey] = self.__convertComplexInList(dictVal.tolist()) if type(dictVal) == np.ndarray else dictVal
                returnVal[regionKey] = tempDict
        else:
            print('The object does not match a conditional')
            return

        return returnVal

    def __filterValType(self, inAttr, inVal):

        # Json dump cannot handle user defined class objects
        if inAttr in self.unacceptedJsonAttributes:
            outVal = self.__objectToDict(inAttr, inVal)

        # Json dump cannot handle numpy arrays
        elif type(inVal) == np.ndarray:
            if inVal.size == 0:
                outVal = []
            else:
                outVal = inVal.tolist()
                # Case 1D list containing unacceptedTypes
                if type(outVal[0]) in self.unacceptedTypeList:
                    outVal = self.__convertComplexInList(outVal)

        # Json dump and load cannot handle complex types
        elif type(inVal) in self.unacceptedTypeList:
            # 'plex_Signature' is used to identify if a list is a destructed complex number or not
            outVal = ['plex_Signature', inVal.real, inVal.imag]

        else:
            outVal = inVal

        return outVal

    def __buildTypeList(self, val):
        # Generate a list of data types before destructing the matrix
        if type(val) not in self.typeList:
            self.typeList.append(type(val))

    def __getType(self, val):
        return str(type(val)).split("'")[1]

    # Encode all attributes that have valid json data types
    def __destruct(self):
        self.encodedAttributes = {}
        for attr, val in self.rawModel.__dict__.items():
            # Matrices included in the set of linear equations were excluded due to computation considerations
            if attr not in self.removedAttributesList:
                self.__buildTypeList(val)
                self.encodedAttributes[attr] = self.__filterValType(attr, val)

    # Rebuild objects that were deconstructed to store in JSON object
    def __construct(self):

        errors = self.encodedAttributes['errorDict']
        matrix = self.encodedAttributes['matrix']

        # ErrorDict reconstruction
        self.encodedAttributes['errorDict'] = TransformedDict.buildFromJson(errors)

        # Matrix reconstruction
        Cnt = 0
        lenKeys = len(matrix)*len(matrix[0])
        constructedMatrix = np.array([type('', (Node,), {}) for _ in range(lenKeys)])
        for row in matrix:
            for nodeInfo in row:
                constructedMatrix[Cnt] = Node.buildFromJson(nodeInfo)
                Cnt += 1
        rawArrShape = self.rawModel.matrix.shape
        constructedMatrix = constructedMatrix.reshape(rawArrShape[0], rawArrShape[1])
        self.encodedAttributes['matrix'] = constructedMatrix

        self.rebuiltModel = Model.buildFromJson(jsonObject=self.encodedAttributes)

    def jsonStoreSolution(self):

        self.__destruct()
        # Data written to file
        if not os.path.isdir(OUTPUT_PATH):
            os.mkdir(OUTPUT_PATH)
        with open(DATA_PATH, 'w') as StoredSolutionData:
            json.dump(self.encodedAttributes, StoredSolutionData)
        # Data read from file
        with open(DATA_PATH) as StoredSolutionData:
            # json.load returns dicts where keys are converted to strings
            self.encodedAttributes = json.load(StoredSolutionData)

        self.__construct()
        if self.rawModel.equals(self.rebuiltModel, self.removedAttributesList):
            self.encoded = True
        else:
            self.rebuiltModel.writeErrorToDict(key='name',
                                               error=Error.buildFromScratch(name='JsonRebuild',
                                                                            description="ERROR - Model Did Not Rebuild Correctly",
                                                                            cause=True))


def main():
    # Efficient to simulate at pixDiv >= 10, but fastest at pixDiv = 2
    pixelDivisions = 5

    slots = 16
    poles = 6
    wt, ws = 6 / 1000, 10 / 1000
    slotpitch = wt + ws
    endTeeth = 2 * (5/3 * wt)
    length = ((slots - 1) * slotpitch + ws) + endTeeth

    pixelSpacing = slotpitch/pixelDivisions
    canvasSpacing = 80

    regionCfg1 = {'hmRegions': {},
                  'mecRegions': {1: 'mec'},
                  'invertY': False}

    choiceRegionCfg = regionCfg1

    # Object for the model design, grid, and matrices
    model = Model.buildFromScratch(slots=slots, poles=poles, length=length,
                                   pixelSpacing=pixelSpacing, canvasSpacing=canvasSpacing,
                                   mecRegions=choiceRegionCfg['mecRegions'],
                                   errorTolerance=1e-14,
                                   # If invertY = False -> [LowerSlot, UpperSlot, Yoke]
                                   # TODO This invertY flips the core MEC region
                                   invertY=choiceRegionCfg['invertY'])

    model.buildGrid()
    model.finalizeGrid(pixelDivisions)

    with timing():
        errorInX = model.finalizeCompute()

    # TODO This invertY inverts the pyplot
    model.updateGrid(errorInX, showAirgapPlot=True, invertY=True, showUnknowns=False)

    # After this point, the json implementations should be used to not branch code direction
    encodeModel = EncoderDecoder(model)
    encodeModel.jsonStoreSolution()

    # TODO The or True is here for convenience but should be removed
    if encodeModel.rebuiltModel.errorDict.isEmpty() or True:
        # iDims (height x width): BenQ = 1440 x 2560, ViewSonic = 1080 x 1920
        # model is only passed in to showModel to show the matrices A and B since they are not stored in the json object
        showModel(encodeModel, model, fieldType='MMF',
                  showGrid=True, showFields=True, showFilter=False, showMatrix=False, showZeros=True,
                  # TODO This invertY inverts the Tkinter Canvas plot
                  numColours=20, dims=[1080, 1920], invertY=False)
        pass
    else:
        print('Resolve errors to show model')

    print('\nBelow are a list of warnings and errors:')
    if encodeModel.rebuiltModel.errorDict:
        encodeModel.rebuiltModel.errorDict.printErrorsByAttr('description')
    else:
        print('   - there are no errors')


if __name__ == '__main__':
    # profile_main()  # To profile the main execution
    main()
