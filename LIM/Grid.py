from builtins import list
from numpy.core._multiarray_umath import ndarray
from LIM.SlotPoleCalculation import*
from collections import deque


class Grid(LimMotor):

    lower_slotsA: ndarray
    upper_slotsA: ndarray
    lower_slotsB: ndarray
    upper_slotsB: ndarray
    lower_slotsC: ndarray
    upper_slotsC: ndarray

    inLower_slotsA: ndarray
    outLower_slotsA: ndarray
    inUpper_slotsA: ndarray
    outUpper_slotsA: ndarray
    inLower_slotsB: ndarray
    outLower_slotsB: ndarray
    inUpper_slotsB: ndarray
    outUpper_slotsB: ndarray
    inLower_slotsC: ndarray
    outLower_slotsC: ndarray
    inUpper_slotsC: ndarray
    outUpper_slotsC: ndarray

    def __init__(self, kwargs):

        super().__init__(kwargs['slots'], kwargs['poles'], kwargs['length'])

        self.invertY = kwargs['invertY']

        xMeshIndexes = kwargs['meshIndexes'][0]
        yMeshIndexes = kwargs['meshIndexes'][1]

        self.Spacing = kwargs['pixelSpacing']
        self.Cspacing = kwargs['canvasSpacing'] / self.H
        self.meshDensity = kwargs['meshDensity']
        self.n = kwargs['n']
        self.hmRegions = kwargs['hmRegions']
        self.mecRegions = kwargs['mecRegions']
        self.hmRegionsIndex = np.zeros(len(self.hmRegions) + 1, dtype=np.int32)
        self.mecRegionsIndex = np.zeros(len(self.mecRegions), dtype=np.int32)

        self.yIndexesBackIron, self.yIndexesBladeRotor, self.yIndexesAirgap = None, None, None
        self.yIndexesLowerSlot, self.yIndexesUpperSlot, self.yIndexesYoke = None, None, None

        self.removeLowerCoilIdxs, self.removeUpperCoilIdxs = [], []
        self.xFirstEdgeNodes, self.xSecondEdgeNodes = [], []
        self.yFirstEdgeNodes, self.ySecondEdgeNodes = [], []

        self.yListPixelsPerRegion = []
        self.yIndexesHM, self.yIndexesMEC = [], []
        self.xBoundaryList, self.yBoundaryList = [], []

        self.yMeshSizes = []

        # X-direction
        self.ppSlotpitch = self.setPixelsPerLength(length=self.slotpitch, minimum=2)
        self.ppAirBuffer = self.setPixelsPerLength(length=self.Airbuffer, minimum=1)
        self.ppTooth = self.setPixelsPerLength(length=self.wt, minimum=1)
        self.ppSlot = self.ppSlotpitch - self.ppTooth
        self.ppEndTooth = self.setPixelsPerLength(length=self.endTeeth, minimum=1)
        # Y-direction
        self.ppVac = self.setPixelsPerLength(length=self.vac, minimum=1)
        self.ppYoke = self.setPixelsPerLength(length=self.hy, minimum=1)
        self.ppAirGap = self.setPixelsPerLength(length=self.g, minimum=1)
        self.ppBladeRotor = self.setPixelsPerLength(length=self.dr, minimum=1)
        self.ppBackIron = self.setPixelsPerLength(length=self.bi, minimum=1)
        self.ppSlotHeight = self.setPixelsPerLength(length=self.hs, minimum=2)

        if self.ppSlotHeight % 2 != 0:
            self.ppSlotHeight += 1

        ppHM = self.ppAirGap + self.ppBladeRotor + self.ppBackIron
        self.ppHeight = self.ppYoke + self.ppSlotHeight
        self.ppH = self.ppHeight + ppHM
        self.ppL = (self.slots - 1) * self.ppSlotpitch + self.ppSlot + 2 * self.ppEndTooth + 2 * self.ppAirBuffer
        self.matrix = np.array([[type('', (Node,), {}) for _ in range(self.ppL)] for _ in range(self.ppHeight + ppHM)])

        self.toothArray, self.coilArray, self.bufferArray = np.zeros((3, 1), dtype=np.int16)
        self.removeLowerCoils, self.removeUpperCoils = np.zeros((2, 1), dtype=np.int16)

        self.xMeshSizes = np.zeros(len(xMeshIndexes), dtype=np.float64)

        # Mesh sizing
        self.fractionSize = (1 / self.meshDensity[0] + 1 / self.meshDensity[1])

        self.mecRegionLength = self.ppHeight * self.ppL

        # Update the hm and mec RegionIndex
        self.setRegionIndices()

        self.xListSpatialDatum = [self.Airbuffer, self.endTeeth] + [self.ws, self.wt] * (self.slots - 1) + [self.ws, self.endTeeth, self.Airbuffer]
        self.xListPixelsPerRegion = [self.ppAirBuffer, self.ppEndTooth] + [self.ppSlot, self.ppTooth] * (self.slots - 1) + [self.ppSlot, self.ppEndTooth, self.ppAirBuffer]
        for Cnt in range(len(self.xMeshSizes)):
            self.xMeshSizes[Cnt] = meshBoundary(self.xListSpatialDatum[Cnt], self.xListPixelsPerRegion[Cnt], self.Spacing, self.fractionSize, sum(xMeshIndexes[Cnt]), self.meshDensity)
            if self.xMeshSizes[Cnt] < 0:
                print('negative x mesh sizes', Cnt)
                return

        offsetList = []
        cnt, offsetLower, offsetUpper = 0, 0, 0
        str_ppSlotHeight = self.getFullRegionDict()['core']['pp'].split(', ')[1]
        for region in self.getFullRegionDict():
            if region.split('_')[0] != 'vac':
                # Set up the y-axis indexes
                for pp in self.getFullRegionDict()[region]['pp'].split(', '):
                    offsetLower = offsetUpper
                    offsetUpper += self.__dict__[pp] // 2 if pp == str_ppSlotHeight else self.__dict__[pp]
                    offsetList.append((offsetLower, offsetUpper))
                for inner_cnt, spatial in enumerate(self.getFullRegionDict()[region]['spatial'].split(', ')):
                    pixelKey = self.getFullRegionDict()[region]['pp'].split(', ')[inner_cnt]
                    pixelVal = self.__dict__[pixelKey] // 2 if pixelKey == str_ppSlotHeight else self.__dict__[pixelKey]
                    spatialVal = self.__dict__[spatial] / 2 if pixelKey == str_ppSlotHeight else self.__dict__[spatial]
                    self.yListPixelsPerRegion.append(pixelVal)
                    self.yMeshSizes.append(meshBoundary(spatialVal, pixelVal, self.Spacing, self.fractionSize,
                                                        sum(yMeshIndexes[cnt]), self.meshDensity))
                    cnt += 1

        # Initialize the y-axis index attributes format: self.yIndexes___
        innerCnt = 0
        for region in self.getFullRegionDict():
            if region.split('_')[0] != 'vac':
                for idx in self.getFullRegionDict()[region]['idx'].split(', '):
                    self.__dict__[idx] = list(range(offsetList[innerCnt][0], offsetList[innerCnt][1]))
                    self.yBoundaryList.append(self.__dict__[idx][-1])
                    if region in self.hmRegions.values():
                        self.yIndexesHM.extend(self.__dict__[idx])
                    elif region in self.mecRegions.values():
                        self.yIndexesMEC.extend(self.__dict__[idx])
                    innerCnt += 1

        # Thrust of the entire integration region
        self.Fx = 0.0
        self.Fy = 0.0

    def buildGrid(self, pixelSpacing, meshIndexes):
        xMeshIndexes = meshIndexes[0]
        yMeshIndexes = meshIndexes[1]

        #  Initialize grid with Nodes
        listOffset = self.ppAirBuffer + self.ppEndTooth
        oldTooth, newTooth, oldSlot, newSlot, slotCount = 0, listOffset, 0, 0, 0
        fullToothArray = []
        for slotCount in range(self.slots - 1):
            oldSlot = newTooth
            newSlot = oldSlot + self.ppSlot
            oldTooth = newSlot
            newTooth = oldTooth + self.ppTooth
            fullToothArray += list(range(oldTooth, newTooth))

        leftEndTooth = list(range(self.ppAirBuffer, listOffset))
        rightEndTooth = list(range(fullToothArray[-1] + self.ppSlot + 1, fullToothArray[-1] + self.ppSlot + self.ppEndTooth + 1))
        self.toothArray = leftEndTooth + fullToothArray + rightEndTooth
        self.coilArray = [x for x in range(self.ppAirBuffer, self.ppL - self.ppAirBuffer) if x not in self.toothArray]
        self.bufferArray = [x for x in list(range(self.ppL)) if x not in self.coilArray and x not in self.toothArray]

        offset = self.ppSlot*math.ceil(self.q)
        intSlots = math.ceil(self.q)*self.poles*3

        # Split slots into respective phases
        temp_upper_slotArray = self.coilArray
        upper_slotArray = temp_upper_slotArray + [None] * self.ppSlot * (intSlots - self.slots)
        lower_slotArray = deque(upper_slotArray)
        lower_slotArray.rotate(-self.windingShift*offset)
        lower_slotArray = list(lower_slotArray)

        upper_slotArrayA, upper_slotArrayB, upper_slotArrayC = [], [], []
        lower_slotArrayA, lower_slotArrayB, lower_slotArrayC = [], [], []
        for threeSlots in range(0, self.slots, 3):
            upper_slotArrayA += upper_slotArray[(threeSlots+1)*offset:(threeSlots+2)*offset]
            upper_slotArrayB += upper_slotArray[threeSlots*offset:(threeSlots+1)*offset]
            upper_slotArrayC += upper_slotArray[(threeSlots+2)*offset:(threeSlots+3)*offset]

            lower_slotArrayA += lower_slotArray[(threeSlots+1)*offset:(threeSlots+2)*offset]
            lower_slotArrayB += lower_slotArray[threeSlots*offset:(threeSlots+1)*offset]
            lower_slotArrayC += lower_slotArray[(threeSlots+2)*offset:(threeSlots+3)*offset]

        self.removeLowerCoils = [0, 1, 2, self.slots-1]
        self.removeUpperCoils = [0, self.slots-1, self.slots-2, self.slots-3]

        for idx in self.removeLowerCoils:
            coilOffset = idx*self.ppSlot
            self.removeLowerCoilIdxs += self.coilArray[coilOffset:coilOffset+self.ppSlot]

        for idx in self.removeUpperCoils:
            coilOffset = idx*self.ppSlot
            self.removeUpperCoilIdxs += self.coilArray[coilOffset:coilOffset+self.ppSlot]

        upper_slotArrayA = [x for x in upper_slotArrayA if x not in self.removeUpperCoilIdxs]
        upper_slotArrayB = [x for x in upper_slotArrayB if x not in self.removeUpperCoilIdxs]
        upper_slotArrayC = [x for x in upper_slotArrayC if x not in self.removeUpperCoilIdxs]
        lower_slotArrayA = [x for x in lower_slotArrayA if x not in self.removeLowerCoilIdxs]
        lower_slotArrayB = [x for x in lower_slotArrayB if x not in self.removeLowerCoilIdxs]
        lower_slotArrayC = [x for x in lower_slotArrayC if x not in self.removeLowerCoilIdxs]

        self.upper_slotsA = np.array(upper_slotArrayA)
        self.lower_slotsA = np.array(lower_slotArrayA)

        self.upper_slotsB = np.array(upper_slotArrayB)
        self.lower_slotsB = np.array(lower_slotArrayB)

        self.lower_slotsC = np.array(lower_slotArrayC)
        self.upper_slotsC = np.array(upper_slotArrayC)

        # Remove None from
        lowerCoilsA_NoneRemoved = list(filter(None, self.lower_slotsA))
        lowerCoilsA = np.split(np.array(lowerCoilsA_NoneRemoved), len(lowerCoilsA_NoneRemoved)//self.ppSlot, axis=0)
        upperCoilsA_NoneRemoved = list(filter(None, self.upper_slotsA))
        upperCoilsA = np.split(np.array(upperCoilsA_NoneRemoved), len(upperCoilsA_NoneRemoved)//self.ppSlot, axis=0)

        lowerCoilsB_NoneRemoved = list(filter(None, self.lower_slotsB))
        lowerCoilsB = np.split(np.array(lowerCoilsB_NoneRemoved), len(lowerCoilsB_NoneRemoved)//self.ppSlot, axis=0)
        upperCoilsB_NoneRemoved = list(filter(None, self.upper_slotsB))
        upperCoilsB = np.split(np.array(upperCoilsB_NoneRemoved), len(upperCoilsB_NoneRemoved)//self.ppSlot, axis=0)

        lowerCoilsC_NoneRemoved = list(filter(None, self.lower_slotsC))
        lowerCoilsC = np.split(np.array(lowerCoilsC_NoneRemoved), len(lowerCoilsC_NoneRemoved)//self.ppSlot, axis=0)
        upperCoilsC_NoneRemoved = list(filter(None, self.upper_slotsC))
        upperCoilsC = np.split(np.array(upperCoilsC_NoneRemoved), len(upperCoilsC_NoneRemoved)//self.ppSlot, axis=0)

        # Sort coils into direction of current (ex, in and out of page)
        self.inLower_slotsA = np.array(lowerCoilsA[1::2])
        self.outLower_slotsA = np.array(lowerCoilsA[::2])
        self.inUpper_slotsA = np.array(upperCoilsA[1::2])
        self.outUpper_slotsA = np.array(upperCoilsA[::2])
        self.inLower_slotsB = np.array(lowerCoilsB[1::2])
        self.outLower_slotsB = np.array(lowerCoilsB[::2])
        self.inUpper_slotsB = np.array(upperCoilsB[1::2])
        self.outUpper_slotsB = np.array(upperCoilsB[::2])
        self.inLower_slotsC = np.array(lowerCoilsC[::2])
        self.outLower_slotsC = np.array(lowerCoilsC[1::2])
        self.inUpper_slotsC = np.array(upperCoilsC[::2])
        self.outUpper_slotsC = np.array(upperCoilsC[1::2])

        # MeshIndexes is a list of 1s and 0s for each boundary for each region to say whether or not
        #  dense meshing is required at the boundary
        # X Mesh Density
        Cnt = 0
        idxOffset = self.xListPixelsPerRegion[Cnt]
        idxLeft, idxRight = 0, idxOffset
        idxList = range(self.ppL)
        for boundary in xMeshIndexes:
            if boundary[0]:  # Left Boundary in the region
                firstIndexes = idxList[idxLeft]
                secondIndexes = idxList[idxLeft + 1]
                self.xFirstEdgeNodes.append(firstIndexes)
                self.xSecondEdgeNodes.append(secondIndexes)
            if boundary[1]:  # Right Boundary in the region
                firstIndexes = idxList[idxRight - 1]
                secondIndexes = idxList[idxRight - 2]
                self.xFirstEdgeNodes.append(firstIndexes)
                self.xSecondEdgeNodes.append(secondIndexes)
            idxLeft += idxOffset
            if Cnt < len(self.xListPixelsPerRegion) - 1:
                idxOffset = self.xListPixelsPerRegion[Cnt+1]
            idxRight += idxOffset
            Cnt += 1

        # Y Mesh Density
        Cnt = 0
        idxOffset = self.yListPixelsPerRegion[Cnt]
        idxLeft, idxRight = 0, idxOffset
        # TODO Here - ppH and yMeshIndexes is not robust
        idxList = range(self.ppH)
        for boundary in yMeshIndexes:
            if boundary[0]:  # Left Boundary in the region
                firstIndexes = idxList[idxLeft]
                secondIndexes = idxList[idxLeft + 1]
                self.yFirstEdgeNodes.append(firstIndexes)
                self.ySecondEdgeNodes.append(secondIndexes)
            if boundary[1]:  # Right Boundary in the region
                firstIndexes = idxList[idxRight - 1]
                secondIndexes = idxList[idxRight - 2]
                self.yFirstEdgeNodes.append(firstIndexes)
                self.ySecondEdgeNodes.append(secondIndexes)
            idxLeft += idxOffset
            if Cnt < len(self.yListPixelsPerRegion) - 1:
                idxOffset = self.yListPixelsPerRegion[Cnt+1]
            idxRight += idxOffset
            Cnt += 1

        self.xBoundaryList = [self.bufferArray[self.ppAirBuffer - 1],
                              self.toothArray[self.ppEndTooth - 1]] + self.coilArray[self.ppSlot - 1::self.ppSlot]\
                              + self.toothArray[self.ppEndTooth + self.ppTooth - 1:-self.ppEndTooth:self.ppTooth] + [self.toothArray[-1],
                              self.bufferArray[-1]]

        a, b = 0, 0
        c, d = 0, 0
        Cnt, yCnt = 0, 0
        # Assign spatial data to the nodes
        while a < self.ppH:

            xCnt = 0
            # Keep track of the y coordinate for each node
            if a in self.yFirstEdgeNodes:
                delY = pixelSpacing / self.meshDensity[0]
            elif a in self.ySecondEdgeNodes:
                delY = pixelSpacing / self.meshDensity[1]
            else:
                delY = self.yMeshSizes[c]

            while b < self.ppL:

                # Keep track of the x coordinate for each node
                if b in self.xFirstEdgeNodes:
                    delX = pixelSpacing / self.meshDensity[0]
                elif b in self.xSecondEdgeNodes:
                    delX = pixelSpacing / self.meshDensity[1]
                else:
                    delX = self.xMeshSizes[d]

                self.matrix[a][b] = Node.buildFromScratch(iIndex=[b, a], iXinfo=[xCnt, delX], iYinfo=[yCnt, delY],
                                                          modelDepth=self.D)

                # Keep track of the x coordinate for each node
                if b in self.xFirstEdgeNodes:
                    xCnt += pixelSpacing / self.meshDensity[0]
                elif b in self.xSecondEdgeNodes:
                    xCnt += pixelSpacing / self.meshDensity[1]
                else:
                    xCnt += delX

                if b in self.xBoundaryList:
                    d += 1

                b += 1
                Cnt += 1
            d = 0

            # Keep track of the y coordinate for each node
            if a in self.yFirstEdgeNodes:
                yCnt += pixelSpacing / self.meshDensity[0]
            elif a in self.ySecondEdgeNodes:
                yCnt += pixelSpacing / self.meshDensity[1]
            else:
                yCnt += delY

            if a in self.yBoundaryList:
                c += 1

            b = 0
            a += 1

        # Assign property data to the nodes
        a, b = 0, 0
        while a < self.ppH:
            while b < self.ppL:
                if a in self.yIndexesYoke and b not in self.bufferArray:
                    self.matrix[a][b].material = 'iron'
                    self.matrix[a][b].ur = self.ur_iron
                    self.matrix[a][b].sigma = self.sigma_iron
                elif a in self.yIndexesAirgap:
                    self.matrix[a][b].material = 'vacuum'
                    self.matrix[a][b].ur = self.ur_air
                    self.matrix[a][b].sigma = self.sigma_air
                elif a in self.yIndexesBladeRotor:
                    self.matrix[a][b].material = 'aluminum'
                    self.matrix[a][b].ur = self.ur_alum
                    self.matrix[a][b].sigma = self.sigma_alum
                elif a in self.yIndexesBackIron:
                    self.matrix[a][b].material = 'iron'
                    self.matrix[a][b].ur = self.ur_iron
                    self.matrix[a][b].sigma = self.sigma_iron
                else:
                    if a in self.yIndexesUpperSlot:
                        aIdx = self.upper_slotsA
                        bIdx = self.upper_slotsB
                        cIdx = self.upper_slotsC
                    elif a in self.yIndexesLowerSlot:
                        aIdx = self.lower_slotsA
                        bIdx = self.lower_slotsB
                        cIdx = self.lower_slotsC
                    else:
                        aIdx = []
                        bIdx = []
                        cIdx = []

                    if b in self.toothArray:
                        self.matrix[a][b].material = 'iron'
                        self.matrix[a][b].ur = self.ur_iron
                        self.matrix[a][b].sigma = self.sigma_iron
                    elif b in self.bufferArray:
                        self.matrix[a][b].material = 'vacuum'
                        self.matrix[a][b].ur = self.ur_air
                        self.matrix[a][b].sigma = self.sigma_air
                    elif b in aIdx:
                        self.matrix[a][b].material = 'copperA'
                        self.matrix[a][b].ur = self.ur_copp
                        self.matrix[a][b].sigma = self.sigma_copp
                    elif b in bIdx:
                        self.matrix[a][b].material = 'copperB'
                        self.matrix[a][b].ur = self.ur_copp
                        self.matrix[a][b].sigma = self.sigma_copp
                    elif b in cIdx:
                        self.matrix[a][b].material = 'copperC'
                        self.matrix[a][b].ur = self.ur_copp
                        self.matrix[a][b].sigma = self.sigma_copp
                    elif b in self.removeLowerCoilIdxs + self.removeUpperCoilIdxs:
                        self.matrix[a][b].material = 'vacuum'
                        self.matrix[a][b].ur = self.ur_air
                        self.matrix[a][b].sigma = self.sigma_air
                    else:
                        self.matrix[a][b].material = ''
                b += 1
            b = 0
            a += 1

    def finalizeGrid(self, pixelDivisions):
        spatialDomainFlag = False

        # Define Indexes
        idxLeftAirBuffer = 0
        idxLeftEndTooth = idxLeftAirBuffer + self.ppAirBuffer
        idxSlot = idxLeftEndTooth + self.ppEndTooth
        idxTooth = idxSlot + self.ppSlot
        idxRightEndTooth = self.ppL - self.ppAirBuffer - self.ppEndTooth
        idxRightAirBuffer = idxRightEndTooth + self.ppEndTooth

        self.checkSpatialMapping(pixelDivisions, spatialDomainFlag,
                                 [idxLeftAirBuffer, idxLeftEndTooth, idxSlot, idxTooth, idxRightEndTooth,
                                  idxRightAirBuffer])

        # Scaling values for MMF-source distribution in section 2.2, equation 18, figure 5
        fraction = 0.5
        doubleBias = self.ppSlotHeight - fraction
        doubleCoilScaling = np.arange(fraction, doubleBias + fraction, 1)
        if self.invertY:
            doubleCoilScaling = np.flip(doubleCoilScaling)

        scalingLower, scalingUpper = 0.0, 0.0
        time_plex = cmath.exp(j_plex * 2 * pi * self.f * self.t)
        turnAreaRatio = self.ppSlot
        i, j = 0, 0
        # TODO Here - ppH, matrix
        while i < self.ppH:
            while j < self.ppL:

                self.matrix[i][j].Rx, self.matrix[i][j].Ry = self.matrix[i][j].getReluctance()

                isCurrentCu = self.matrix[i][j].material[:-1] == 'copper'
                if i in self.yIndexesLowerSlot and j in self.coilArray:
                    if j in self.lower_slotsA:
                        angle_plex = cmath.exp(0)
                    elif j in self.lower_slotsB:
                        angle_plex = cmath.exp(-j_plex * pi * 2 / 3)
                    elif j in self.lower_slotsC:
                        angle_plex = cmath.exp(j_plex * pi * 2 / 3)
                    else:
                        angle_plex = 0.0

                    # Set the scaling factor for MMF in equation 18
                    if isCurrentCu:
                        index_ = self.yIndexesLowerSlot.index(i)
                        if self.invertY:
                            index_ += len(doubleCoilScaling) // 2
                        scalingLower = doubleCoilScaling[index_]
                    else:
                        scalingLower = 0.0

                elif i in self.yIndexesUpperSlot and j in self.coilArray:
                    if j in self.upper_slotsA:
                        angle_plex = cmath.exp(0)
                    elif j in self.upper_slotsB:
                        angle_plex = cmath.exp(-j_plex * pi * 2 / 3)
                    elif j in self.upper_slotsC:
                        angle_plex = cmath.exp(j_plex * pi * 2 / 3)
                    else:
                        angle_plex = 0.0

                    # Set the scaling factor for MMF in equation 18
                    # 2 coils in slot
                    yLowerCoilIdx = i + self.ppSlotHeight // 2 if self.invertY else i - self.ppSlotHeight // 2
                    isLowerCu = self.matrix[yLowerCoilIdx][j].material[:-1] == 'copper'
                    if isCurrentCu and isLowerCu:
                        index_ = self.yIndexesUpperSlot.index(i)
                        if not self.invertY:
                            index_ += len(doubleCoilScaling) // 2
                        scalingUpper = doubleCoilScaling[index_]
                    # coil in upper slot only
                    elif isCurrentCu and not isLowerCu:
                        index_ = self.yIndexesUpperSlot.index(i)
                        if self.invertY:
                            index_ -= len(doubleCoilScaling) // 2
                        scalingUpper = doubleCoilScaling[index_]
                    else:
                        scalingUpper = 0.0

                else:
                    angle_plex = 0.0
                    scalingLower = 0.0
                    scalingUpper = 0.0

                self.matrix[i][j].Iph = self.Ip * angle_plex * time_plex

                # Lower slots only
                if i in self.yIndexesLowerSlot and j in self.coilArray:
                    if j in self.inLower_slotsA or j in self.inLower_slotsB or j in self.inLower_slotsC:
                        inOutCoeffMMF = -1
                    elif j in self.outLower_slotsA or j in self.outLower_slotsB or j in self.outLower_slotsC:
                        inOutCoeffMMF = 1
                    else:
                        if j not in self.removeLowerCoilIdxs:
                            print('Shouldnt happen')
                        inOutCoeffMMF = 0

                    self.matrix[i][j].MMF = inOutCoeffMMF * scalingLower * self.N * self.matrix[i][j].Iph / (2 * turnAreaRatio)

                # Upper slots only
                elif i in self.yIndexesUpperSlot and j in self.coilArray:
                    if j in self.inUpper_slotsA or j in self.inUpper_slotsB or j in self.inUpper_slotsC:
                        inOutCoeffMMF = -1
                    elif j in self.outUpper_slotsA or j in self.outUpper_slotsB or j in self.outUpper_slotsC:
                        inOutCoeffMMF = 1
                    else:
                        inOutCoeffMMF = 0

                    self.matrix[i][j].MMF = inOutCoeffMMF * scalingUpper * self.N * self.matrix[i][j].Iph / (2 * turnAreaRatio)

                # Stator teeth
                else:
                    self.matrix[i][j].MMF = 0.0

                j += 1
            j = 0
            i += 1

    def setPixelsPerLength(self, length, minimum):

        pixels = round(length / self.Spacing)
        return minimum if pixels < minimum else pixels

    def setRegionIndices(self):

        # Create the starting index for each region in the columns of matrix A
        regionIndex, hmCount, mecCount = 0, 0, 0
        for count in range(1, len(self.mecRegions) + len(self.hmRegions) + 2):
            if count in self.hmRegions:
                # Dirichlet boundaries have half the unknown coefficients
                if self.hmRegions[count].split('_')[0] == 'vac':
                    self.hmRegionsIndex[hmCount] = regionIndex
                    regionIndex += len(self.n)

                else:
                    self.hmRegionsIndex[hmCount] = regionIndex
                    regionIndex += 2 * len(self.n)
                hmCount += 1

            elif count in self.mecRegions:
                self.mecRegionsIndex[mecCount] = regionIndex
                regionIndex += self.mecRegionLength
                mecCount += 1

            elif count == list(self.hmRegions)[-1] + 1:
                self.hmRegionsIndex[hmCount] = regionIndex

            else:
                self.writeErrorToDict(key='name',
                                      error=Error.buildFromScratch(name='gridRegion',
                                                                   description='ERROR - Error in the iGrid regions list',
                                                                   cause=True))

    def getFullRegionDict(self):
        config = configparser.ConfigParser()
        config.read('Properties.ini')
        regDict = {}
        for key in range(list(self.hmRegions.keys())[0], list(self.hmRegions.keys())[-1] + 1):
            if key in self.hmRegions:
                hmType = self.hmRegions[key]
                regDict[hmType] = {'pp': config.get('PP', hmType),
                                   'bc': config.get('BC', hmType),
                                   'idx': config.get('Y_IDX', hmType),
                                   'spatial': config.get('SPATIAL', hmType)}

            elif key in self.mecRegions:
                mecType = self.mecRegions[key]
                regDict[mecType] = {'pp': config.get('PP', mecType),
                                    'bc': config.get('BC', mecType),
                                    'idx': config.get('Y_IDX', mecType),
                                    'spatial': config.get('SPATIAL', mecType)}
                # We need to reverse the Properties.ini file
                if self.invertY:
                    for each in regDict[mecType]:
                        stringList = ''
                        temp = regDict[mecType][each].split(', ')
                        temp.reverse()
                        for cnt, string in enumerate(temp):
                            if cnt == 0:
                                stringList += string
                            else:
                                stringList += f', {string}'
                        regDict[mecType][each] = stringList

            else:
                self.writeErrorToDict(key='name',
                                      error=Error.buildFromScratch(name='InputRegions',
                                                                   description='ERROR - The input regions overlap between mec and hm',
                                                                   cause=True))
        return regDict

    def getLastAndNextRegionName(self, name):
        previous = list(self.getFullRegionDict())[list(self.getFullRegionDict()).index(name)-1]
        next_ = list(self.getFullRegionDict())[list(self.getFullRegionDict()).index(name)+1]

        return previous, next_

    # This function is written to catch any errors in the mapping between canvas and space for both x and y coordinates
    def checkSpatialMapping(self, pixelDivision, spatialDomainFlag, iIdxs):

        iIdxLeftAirBuffer, iIdxLeftEndTooth, iIdxSlot, iIdxTooth, iIdxRightEndTooth, iIdxRightAirBuffer = iIdxs

        # The difference between expected and actual is rounded due to quantization error
        #  in the 64 bit floating point arithmetic

        # X direction Checks

        yIdx = 0
        # Check grid spacing to slotpitch
        if round(self.slotpitch - self.Spacing * pixelDivision, 12) != 0:
            print(f'flag - iGrid spacing: {self.slotpitch - self.Spacing * pixelDivision}')
            spatialDomainFlag = True
        # Check slotpitch
        if round(self.slotpitch - (self.matrix[yIdx][iIdxTooth + self.ppTooth].x - self.matrix[yIdx][iIdxSlot].x), 12) != 0:
            print(f'flag - slotpitch: {self.slotpitch - (self.matrix[yIdx][iIdxTooth + self.ppTooth].x - self.matrix[yIdx][iIdxSlot].x)}')
            spatialDomainFlag = True
        # Check slot width
        if round(self.ws - (self.matrix[yIdx][iIdxTooth].x - self.matrix[yIdx][iIdxSlot].x), 12) != 0:
            print(f'flag - slots: {self.ws - (self.matrix[yIdx][iIdxTooth].x - self.matrix[yIdx][iIdxSlot].x)}')
            spatialDomainFlag = True
        # Check tooth width
        if round(self.wt - (self.matrix[yIdx][iIdxTooth + self.ppTooth].x - self.matrix[yIdx][iIdxTooth].x), 12) != 0:
            print(f'flag - teeth: {self.wt - (self.matrix[yIdx][iIdxTooth + self.ppTooth].x - self.matrix[yIdx][iIdxTooth].x)}')
            spatialDomainFlag = True
        # Check left end tooth
        if round(self.endTeeth - (self.matrix[yIdx][iIdxSlot].x - self.matrix[yIdx][iIdxLeftEndTooth].x), 12) != 0:
            print(f'flag - left end tooth: {self.endTeeth - (self.matrix[yIdx][iIdxSlot].x - self.matrix[yIdx][iIdxLeftEndTooth].x)}')
            spatialDomainFlag = True
        # Check right end tooth
        if round(self.endTeeth - (self.matrix[yIdx][iIdxRightAirBuffer].x - self.matrix[yIdx][iIdxRightEndTooth].x), 12) != 0:
            print(f'flag - right end tooth: {self.endTeeth - (self.matrix[yIdx][iIdxRightAirBuffer].x - self.matrix[yIdx][iIdxRightEndTooth].x)}')
            spatialDomainFlag = True
        # Check left air buffer
        if round(self.Airbuffer - (self.matrix[yIdx][iIdxLeftEndTooth].x - self.matrix[yIdx][iIdxLeftAirBuffer].x), 12) != 0:
            print(f'flag - left air buffer: {self.Airbuffer - (self.matrix[yIdx][iIdxLeftEndTooth].x - self.matrix[yIdx][iIdxLeftAirBuffer].x)}')
            spatialDomainFlag = True
        # Check right air buffer
        if round(self.Airbuffer - (self.matrix[yIdx][-1].x + self.matrix[yIdx][-1].lx - self.matrix[yIdx][iIdxRightAirBuffer].x), 12) != 0:
            print(f'flag - right air buffer: {self.Airbuffer - (self.matrix[yIdx][-1].x + self.matrix[yIdx][-1].lx - self.matrix[yIdx][iIdxRightAirBuffer].x)}')
            spatialDomainFlag = True

        # Y direction Checks

        xIdx = 0
        # Check Blade Rotor
        diffBladeRotorDims = self.dr - (self.matrix[self.yIndexesBladeRotor[-1]+1][xIdx].y - self.matrix[self.yIndexesBladeRotor[0]][xIdx].y)
        if round(diffBladeRotorDims, 12) != 0:
            print(f'flag - blade rotor: {diffBladeRotorDims}')
            spatialDomainFlag = True
        # Check Air Gap
        diffAirGapDims = self.g - (self.matrix[self.yIndexesAirgap[-1]+1][xIdx].y - self.matrix[self.yIndexesAirgap[0]][xIdx].y)
        if round(diffAirGapDims, 12) != 0:
            print(f'flag - air gap: {diffAirGapDims}')
            spatialDomainFlag = True
        # Check Lower Slot Height
        diffLowerSlotHeightDims = self.hs/2 - (self.matrix[self.yIndexesLowerSlot[-1]+1][xIdx].y - self.matrix[self.yIndexesLowerSlot[0]][xIdx].y)
        if round(diffLowerSlotHeightDims, 12) != 0:
            print(f'flag - slot height: {diffLowerSlotHeightDims}')
            spatialDomainFlag = True
        # Check Upper Slot Height
        diffUpperSlotHeightDims = self.hs/2 - (self.matrix[self.yIndexesUpperSlot[-1]+1][xIdx].y - self.matrix[self.yIndexesUpperSlot[0]][xIdx].y)
        if round(diffUpperSlotHeightDims, 12) != 0:
            print(f'flag - slot height: {diffUpperSlotHeightDims}')
            spatialDomainFlag = True
        # Check Yoke and Back Iron since they can be the last region next to upperVac
        isYokeAtUpperVac = list(self.getFullRegionDict())[1:-1].index('core') == len(list(self.getFullRegionDict())[1:-1]) - 1
        # Check if yoke is next to upperVac, otherwise index out of bounds will occur
        if isYokeAtUpperVac and not self.invertY:
            yokeY = self.matrix[self.yIndexesYoke[-1]][xIdx].y + self.matrix[self.yIndexesYoke[-1]][xIdx].ly - self.matrix[self.yIndexesYoke[0]][xIdx].y
        else:
            yokeY = self.matrix[self.yIndexesYoke[-1]+1][xIdx].y - self.matrix[self.yIndexesYoke[0]][xIdx].y
        diffYokeDims = self.hy - yokeY
        if round(diffYokeDims, 12) != 0:
            print(f'flag - yoke: {diffYokeDims}')
            spatialDomainFlag = True

        isBIAtUpperVac = list(self.getFullRegionDict())[1:-1].index('bi') == len(list(self.getFullRegionDict())[1:-1]) - 1
        # Check if yoke is next to upperVac, otherwise index out of bounds will occur
        if isBIAtUpperVac:
            biY = self.matrix[self.yIndexesBackIron[-1]][xIdx].y + self.matrix[self.yIndexesBackIron[-1]][xIdx].ly - self.matrix[self.yIndexesBackIron[0]][xIdx].y
        else:
            biY = self.matrix[self.yIndexesBackIron[-1]+1][xIdx].y - self.matrix[self.yIndexesBackIron[0]][xIdx].y
        diffBackIronDims = self.bi - biY
        if round(diffBackIronDims, 12) != 0:
            print(f'flag - blade rotor: {diffBackIronDims}')
            spatialDomainFlag = True

        self.writeErrorToDict(key='name',
                              error=Error.buildFromScratch(name='domainMapping',
                                                           description='ERROR - The spatial domain does not match with the canvas domain',
                                                           cause=spatialDomainFlag))


class Node(object):
    def __init__(self, kwargs, buildFromJson=False):

        if buildFromJson:
            for attr_key in kwargs:
                if type(kwargs[attr_key]) == list and kwargs[attr_key][0] == 'plex_Signature':
                    self.__dict__[attr_key] = rebuildPlex(kwargs[attr_key])
                else:
                    self.__dict__[attr_key] = kwargs[attr_key]
            return

        self.xIndex = kwargs['iIndex'][0]
        self.yIndex = kwargs['iIndex'][1]

        # Initialize dense meshing near slot and teeth edges in x direction
        self.x = kwargs['iXinfo'][0]
        self.lx = kwargs['iXinfo'][1]

        # Initialize dense meshing near slot and teeth edges in y direction
        self.y = kwargs['iYinfo'][0]
        self.ly = kwargs['iYinfo'][1]

        # Cross sectional area
        self.Szy = self.ly*kwargs['modelDepth']
        self.Sxz = self.lx*kwargs['modelDepth']

        self.xCenter = self.x + self.lx / 2
        self.yCenter = self.y + self.ly / 2

        # Node properties
        self.ur = 0.0
        self.sigma = 0.0
        self.material = ''
        self.colour = ''

        # Potential
        self.Yk = np.cdouble(0.0)

        # Thrust
        self.Fx, self.Fy, self.F = np.zeros(3, dtype=np.cdouble)

        # B field
        self.Bx, self.By, self.B, self.BxLower, self.ByLower, self.B_Lower = np.zeros(6, dtype=np.cdouble)

        # Flux
        self.phiXp, self.phiXn, self.phiYp, self.phiYn, self.phiX, self.phiY, self.phi = np.zeros(7, dtype=np.cdouble)
        self.phiError = np.cdouble(0.0)

        # Current
        self.Iph = np.cdouble(0.0)  # phase

        # MMF
        self.MMF = np.cdouble(0.0)  # AmpereTurns

        # Reluctance
        self.Rx, self.Ry = np.zeros(2, dtype=np.float64)

    @classmethod
    def buildFromScratch(cls, **kwargs):
        return cls(kwargs=kwargs)

    @classmethod
    def buildFromJson(cls, jsonObject):
        return cls(kwargs=jsonObject, buildFromJson=True)

    def __eq__(self, otherObject):
        if not isinstance(otherObject, Node):
            # don't attempt to compare against unrelated types
            return NotImplemented
        # If the objects are the same then set the IDs to be equal
        elif self.__dict__.items() == otherObject.__dict__.items():
            for attr, val in otherObject.__dict__.items():
                return self.__dict__[attr] == otherObject.__dict__[attr]
        # The objects are not the same
        else:
            pass

    def drawNode(self, canvasSpacing, overRideColour, c, nodeWidth):

        x_old = self.x*canvasSpacing
        x_new = x_old + self.lx*canvasSpacing
        y_old = self.y*canvasSpacing
        y_new = y_old + self.ly*canvasSpacing

        if overRideColour:
            fillColour = overRideColour
        else:
            idx = np.where(matList == self.material)
            if len(idx[0] == 1):
                matIdx = idx[0][0]
                self.colour = matList[matIdx][1]
            else:
                self.colour = 'orange'
            fillColour = self.colour

        c.create_rectangle(x_old, y_old, x_new, y_new, width=nodeWidth, fill=fillColour)

    def getReluctance(self, vacHeight=0, isVac=False):

        ResX = self.lx / (2 * uo * self.ur * self.Szy)
        if isVac:
            ResY = vacHeight / (2 * uo * self.ur * self.Sxz)
        else:
            ResY = self.ly / (2 * uo * self.ur * self.Sxz)

        return ResX, ResY


class Region(object):
    def __init__(self, kwargs, buildFromJson=False):

        if buildFromJson:
            for key in kwargs:
                if key in ['an', 'bn']:
                    if type(kwargs[key]) == np.ndarray:
                        valList = [rebuildPlex(index) for index in kwargs[key]]
                        valArray = np.array(valList)
                        self.__dict__[key] = valArray
                    else:
                        self.__dict__[key] = kwargs[key]
                else:
                    self.__dict__[key] = kwargs[key]
            return

        self.type = kwargs['type']
        self.an = kwargs['an']
        self.bn = kwargs['bn']

    @classmethod
    def buildFromScratch(cls, **kwargs):
        return cls(kwargs=kwargs)

    @classmethod
    def rebuildFromJson(cls, jsonObject):
        return cls(kwargs=jsonObject, buildFromJson=True)


def meshBoundary(spatial, pixels, spacing, fraction, numBoundaries, meshDensity):

    meshSize = (spatial - numBoundaries*fraction*spacing) / (pixels - len(meshDensity)*numBoundaries)

    return meshSize
