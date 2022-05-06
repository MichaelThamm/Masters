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


# Problem is the object that Test will inherit from
class Test(Problem):

    def __init__(self, iN):
        self.n = iN
        # The numbers indicate: #inputs, #objectives, #constraints
        super(Test, self).__init__(2, 2, 1)
        # Constrain the range and type for each input
        self.types[:] = [Integer(12, 50), Integer(12, 50)]
        # Constrain every input. This works in unison with with constraints in the evaluate method
        # Ex) Slots >= Poles becomes Slots - Poles >= 0 so init constraint becomes ">=0" and evaluate constraint becomes Slots - Poles
        self.constraints[:] = ">=0"
        # Choose which objective to maximize and minimize
        self.directions[:] = [self.MINIMIZE, self.MAXIMIZE]

    def evaluate(self, solution):
        # TODO To get this to work you need to uncomment the robust ratios in SlotPoleCalculation
        slots = solution.variables[0]
        poles = solution.variables[1]

        q = slots / poles / 3
        if slots > poles and slots > 6 and poles % 2 == 0 and q % 1 == 0 and q != 0:

            pixelDivisions = 5
            lowDiscrete = 50
            n = range(-lowDiscrete, lowDiscrete + 1)
            n = np.delete(n, len(n) // 2, 0)
            wt, ws = 6 / 1000, 10 / 1000
            slotpitch = wt + ws
            endTeeth = 2 * (5 / 3 * wt)
            length = ((slots - 1) * slotpitch + ws) + endTeeth

            meshDensity = np.array([4, 2])
            xMeshIndexes = [[0, 0]] + [[0, 0]] + [[0, 0], [0, 0]] * (slots - 1) + [[0, 0]] + [[0, 0]] + [[0, 0]]
            yMeshIndexes = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
            pixelSpacing = slotpitch / pixelDivisions
            canvasSpacing = 80

            regionCfg3 = {'hmRegions': {1: 'vac_lower', 2: 'bi', 3: 'dr', 5: 'vac_upper'},
                          'mecRegions': {4: 'mec'},
                          'invertY': False}

            choiceRegionCfg = regionCfg3

            # Object for the model design, grid, and matrices
            model = Model.buildFromScratch(slots=slots, poles=poles, length=length, n=n,
                                           pixelSpacing=pixelSpacing, canvasSpacing=canvasSpacing,
                                           meshDensity=meshDensity, meshIndexes=[xMeshIndexes, yMeshIndexes],
                                           hmRegions=
                                           choiceRegionCfg['hmRegions'],
                                           mecRegions=
                                           choiceRegionCfg['mecRegions'],
                                           errorTolerance=1e-15,
                                           # If invertY = False -> [LowerSlot, UpperSlot, Yoke]
                                           # TODO This invertY flips the core MEC region
                                           invertY=choiceRegionCfg['invertY'])

            model.buildGrid(pixelSpacing, [xMeshIndexes, yMeshIndexes])
            model.finalizeGrid(pixelDivisions)
            errorInX = model.finalizeCompute()
            model.updateGrid(errorInX, showAirgapPlot=False, invertY=True, showUnknowns=False)

            mass_fitness = model.MassTot
            thrust_fitness = model.Fx
        else:
            mass_fitness = math.inf
            thrust_fitness = 0

        print('solution: ', slots, poles, mass_fitness, thrust_fitness)

        solution.objectives[:] = [mass_fitness, thrust_fitness]
        solution.constraints[:] = [slots - poles]


# Break apart the grid.matrix array

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


def platypus(n, run=False):
    if not run:
        return

    with timing():
        algorithm = NSGAII(Test(n))
        algorithm.run(5000)

    # # TODO Platypus has parallelization built in instead of using Jit
    # # TODO Look into changing population
    feasible_solutions = [s for s in algorithm.result if s.feasible]
    # # TODO Use the debugging tool to see the attributes of algorithm.population
    plotResults = []
    cnt = 1
    for results in feasible_solutions:
        slotpoles = [algorithm.problem.types[0].decode(results.variables[0]), algorithm.problem.types[0].decode(results.variables[1])]
        objectives = [results.objectives[0], results.objectives[1]]
        # TODO Think of a good way to plot results
        print(slotpoles, objectives)
        plotResults.append((slotpoles, objectives))
        cnt += 1

    fig, axs = plt.subplots(2)
    fig.suptitle('Mass and Thrust')
    axs[0].scatter(range(1, cnt), [x[1, 0] for x in plotResults])
    axs[0].set_title('Mass')
    axs[1].scatter(range(1, cnt), [x[1, 1] for x in plotResults])
    axs[1].set_title('Thrust')
    axs[2].scatter(range(1, cnt), [x[0, 0] for x in plotResults])
    axs[2].set_title('mass obj')
    axs[3].scatter(range(1, cnt), [x[0, 1] for x in plotResults])
    axs[3].set_title('thrust obj')
    plt.scatter(range(1, cnt), [x[1, 1] for x in plotResults])
    # plt.scatter(range(1, cnt), [x[1, 1] for x in plotResults])
    plt.xlabel('Generations')
    plt.ylabel('Thrust Objective Value')
    plt.show()


def profile_main():

    import cProfile, pstats, io

    prof = cProfile.Profile()
    prof = prof.runctx("main()", globals(), locals())

    stream = io.StringIO()

    stats = pstats.Stats(prof, stream=stream)
    stats.sort_stats("time")  # or cumulative
    stats.print_stats(80)  # 80 = how many to print

    # The rest is optional.
    # stats.print_callees()
    # stats.print_callers()

    # logging.info("Profile data:\n%s", stream.getvalue())

    f = open(os.path.join(OUTPUT_PATH, 'profile.txt'), 'a')
    f.write(stream.getvalue())
    f.close()


def main():
    # Efficient to simulate at pixDiv >= 10, but fastest at pixDiv = 2
    pixelDivisions = 5

    slots = 16
    poles = 6
    wt, ws = 6 / 1000, 10 / 1000
    slotpitch = wt + ws
    endTeeth = 2 * (5/3 * wt)
    length = ((slots - 1) * slotpitch + ws) + endTeeth

    # platypus(n, run=False)

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
