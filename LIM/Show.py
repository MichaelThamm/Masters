from LIM.Compute import *
from LIM.CanvasModel import CanvasInFrame
from tkinter import *
import matplotlib as mpl
import matplotlib.pyplot as plt


# noinspection PyUnresolvedReferences
def colorFader(c1, c2, mix=0):  # fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)

    c1 = np.array(mpl.colors.to_rgb(c1))
    c2 = np.array(mpl.colors.to_rgb(c2))

    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)


def myColourNumber(fieldScale, val):

    return val if np.isinf(val) else min(fieldScale, key=lambda x: abs(x - val))


def determineColour(jsonObject, iI, iJ, field, highlightZeroValsInField):

    fieldType, iFieldsScale, iStoColours, iPosInf, iNegInf = field
    if str(type(jsonObject.rebuiltModel.matrix[iI, iJ].__dict__[fieldType])).split("'")[1] in jsonObject.unacceptedTypeList:
        myNumber = myColourNumber(iFieldsScale, jsonObject.rebuiltModel.matrix[iI, iJ].__dict__[fieldType].real)
        valEqZero = True if (jsonObject.rebuiltModel.matrix[iI, iJ].__dict__[fieldType].real == 0 and highlightZeroValsInField) else False
    else:
        myNumber = myColourNumber(iFieldsScale, jsonObject.rebuiltModel.matrix[iI, iJ].__dict__[fieldType])
        valEqZero = True if (jsonObject.rebuiltModel.matrix[iI, iJ].__dict__[fieldType] == 0 and highlightZeroValsInField) else False

    # noinspection PyUnboundLocalVariable
    colorScaleIndex = np.where(iFieldsScale == myNumber) if not np.isinf(myNumber) else myNumber
    if valEqZero:
        oOverRideColour = matList[-1][1]
    elif np.isinf(myNumber):
        oOverRideColour = '#000000'
    else:
        oOverRideColour = iStoColours[colorScaleIndex[0][0]]

    return oOverRideColour


def minMaxField(jsonObject, attName, filtered, showFilter):

    model = jsonObject.rebuiltModel
    iFilteredRows, iFilteredRowCols = filtered
    if showFilter:
        tFiltered = [[model.matrix[x.yIndex, x.xIndex] for x in y if x.yIndex in iFilteredRows or (x.yIndex, x.xIndex) in iFilteredRowCols] for y in model.matrix]

    else:
        tFiltered = model.matrix

    tFilteredNoEmpties = list(filter(lambda x: True if x != [] else False, tFiltered))

    attrFieldList = [x.__dict__[attName].real if type(x.__dict__[attName]) in jsonObject.unacceptedTypeList else x.__dict__[attName] for y in tFilteredNoEmpties for x in y]
    filteredAttrFieldList = list(filter(lambda x: not np.isinf(x), attrFieldList))
    maxScale = max(filteredAttrFieldList)
    minScale = min(filteredAttrFieldList)

    return minScale, maxScale


def combineFilterList(pixelsPer, unfilteredRows, unfilteredRowCols):

    ppH, ppL = pixelsPer
    oFilteredRows = []
    oFilteredRowCols = []

    # List of row indexes for keeping rows from 0 to ppH
    for a in range(len(unfilteredRows)):
        oFilteredRows += unfilteredRows[a]

    # List of (row, col) indexes for keeping nodes in mesh
    for a in range(len(unfilteredRowCols)):
        for y in range(ppH):
            for x in range(ppL):
                if y in unfilteredRowCols[a][0] and x in unfilteredRowCols[a][1]:
                    oFilteredRowCols += [(y, x)]

    return oFilteredRows, oFilteredRowCols


def genCanvasMatrix(matrix, canvasRowRegions, bShowA=False, bShowB=False):

    colourSet = ['#DAF7A6', '#FFC300', '#FF5733', '#C70039', '#900C3F', '#581845']
    colourMem = 'orange'
    iterColourSet = iter(colourSet)
    oY_new = 0

    for i in range(len(matrix)):

        oY_old = oY_new
        oY_new = oY_old + 1
        oX_new = 0

        if i in canvasRowRegions:
            colourMem = next(iterColourSet)

        if bShowA:

            # If iMatrix is 2D
            for j in range(len(matrix[i])):
                oX_old = oX_new
                oX_new = oX_old + 1

                if matrix[i, j] == 0:
                    oFillColour = 'empty'

                else:
                    oFillColour = colourMem

                yield oX_old, oY_old, oX_new, oY_new, oFillColour

        if bShowB:

            # If iMatrix is 1D
            oX_old = 0
            oX_new = 10

            if matrix[i] == 0:
                oFillColour = 'empty'

            else:
                oFillColour = colourMem

            yield oX_old, oY_old, oX_new, oY_new, oFillColour


def visualizeMatrix(dims, model, ogModel, bShowA=False, bShowB=False):

    if bShowA and bShowB:
        print('Please init visualizeMatrix function for matrix A and B separately')
        return

    elif not bShowA and not bShowB:
        print('Neither A or B matrix was chosen to be visualized')
        return

    lineLength = len(ogModel.matrixA)
    cMat = CanvasInFrame(height=dims[0]//2, width=dims[1]//2, bg='gray30')
    cMat.root.title('Matrix Visualization')
    if bShowA:
        # Grid row region lines
        for val in model.canvasRowRegIdxs:
            cMat.canvas.create_line(0, val, lineLength, val, fill='blue',
                                    dash=(4, 2))  # Horizontal Lines

        # Grid col region lines
        for val in model.canvasColRegIdxs:
            cMat.canvas.create_line(val, 0, val, lineLength, fill='blue',
                                    dash=(4, 2))  # Vertical Lines

        # Grid MEC-HM boundary region lines
        for val in model.mecCanvasRegIdxs:
            cMat.canvas.create_line(0, val, lineLength, val, fill='red',
                                    dash=(4, 2))  # Horizontal Lines
            cMat.canvas.create_line(val, 0, val, lineLength, fill='red',
                                    dash=(4, 2))  # Vertical Lines

        matrixCanvasGenerator = genCanvasMatrix(ogModel.matrixA, model.canvasRowRegIdxs, bShowA=bShowA, bShowB=bShowB)

    elif bShowB:
        # Grid row region lines
        for val in model.canvasRowRegIdxs:
            cMat.canvas.create_line(0, val, dims[1]//2, val, fill='blue',
                                    dash=(4, 2))  # Horizontal Lines

        # Grid MEC-HM boundary region lines
        for val in model.mecCanvasRegIdxs:
            cMat.canvas.create_line(0, val, dims[1]//2, val, fill='red',
                                    dash=(4, 2))  # Horizontal Lines

        matrixCanvasGenerator = genCanvasMatrix(ogModel.matrixB, model.canvasRowRegIdxs, bShowA=bShowA, bShowB=bShowB)

    else:
        matrixCanvasGenerator = None

    # Colour nodes that have a non-zero value
    for i in matrixCanvasGenerator:
        x_old, y_old, x_new, y_new, fillColour = i
        if fillColour == 'empty':
            pass
        else:
            cMat.canvas.create_rectangle(x_old, y_old, x_new, y_new, width=0, fill=fillColour)

    cMat.pack(side=LEFT, expand=True, fill=BOTH)
    cMat.mainloop()


def showModel(jsonObject, ogModel, canvasCfg, numColours, dims, invertY):

    invertCoeff = -1 if invertY else 1

    # Create the grid canvas to display the grid mesh
    if canvasCfg["showGrid"]:

        cGrid = CanvasInFrame(height=dims[0], width=dims[1], bg='gray30')
        i, j = 0, 0
        while i < jsonObject.rebuiltModel.matrix.shape[0]:
            while j < jsonObject.rebuiltModel.matrix.shape[1]:
                jsonObject.rebuiltModel.matrix[i, j].drawNode(canvasSpacing=jsonObject.rebuiltModel.Cspacing,
                                                              overRideColour=False, c=cGrid.canvas, nodeWidth=1)
                j += 1
            j = 0
            i += 1

        cGrid.canvas.scale("all", 0, 0, 1, invertCoeff)
        cGrid.canvas.configure(scrollregion=cGrid.canvas.bbox("all"))
        cGrid.root.mainloop()

    # Color nodes based on node values
    if canvasCfg["showFields"] or canvasCfg["showFilter"]:

        cFields = CanvasInFrame(height=dims[0], width=dims[1], bg='gray30')

        c1 = '#FFF888'  # Yellow Positive Limit
        c2 = '#700000'  # Dark Red Negative Limit
        cPosInf = '#00FFFF'  # BabyBlue Positive Infinity
        cNegInf = '#9EFE4C'  # Green Negative Infinity
        stoColours = ["empty"]*(numColours + 1)
        for x in range(numColours + 1):
            stoColours[x] = colorFader(c1, c2, x / numColours)

        # Max and Min values for normalizing the color scales for field analysis
        keepRows = [[]]*1
        keepRowColsUnfiltered = [[]]*2

        # [row]
        # Rule 1
        keepRows[0] = jsonObject.rebuiltModel.yIndexesAirGap

        # [row, col] - Make sure to put the rules in order of ascending rows or the list wont be sorted (shouldnt matter)
        # Rule 1
        keepRowColsUnfiltered[0] = [jsonObject.rebuiltModel.yIndexesYoke, jsonObject.rebuiltModel.toothArray + jsonObject.rebuiltModel.slotArray]
        # Rule 2
        keepRowColsUnfiltered[1] = [jsonObject.rebuiltModel.yIndexesLowerSlot + jsonObject.rebuiltModel.yIndexesUpperSlot, jsonObject.rebuiltModel.toothArray]

        filteredRows, filteredRowCols = combineFilterList([jsonObject.rebuiltModel.ppH, jsonObject.rebuiltModel.ppL], keepRows, keepRowColsUnfiltered)

        minScale, maxScale = minMaxField(jsonObject, canvasCfg["fieldType"], [filteredRows, filteredRowCols], canvasCfg["showFilter"])
        normScale = (maxScale - minScale) / (numColours - 1)

        if [minScale, maxScale, normScale] == [0, 0, 0]:

            jsonObject.rebuiltModel.writeErrorToDict(key='name',
                                   error=Error.buildFromScratch(name='emptyField',
                                                                description=f'Field Analysis Error. All values are zero! Type: {canvasCfg["fieldType"]}',
                                                                cause=True))

        # Create fields canvas to display the selected field result on the mesh
        else:
            fieldsScale = np.arange(minScale, maxScale + normScale, normScale)
            colorScaleIndex = np.where(fieldsScale == fieldsScale[0])

            cFields.canvas.create_text(400, 1000, font="Purisa", text=f"Debug (Max, Min): ({maxScale}, {minScale}) colour: ({stoColours[-1]}, {stoColours[colorScaleIndex[0][0]]}) Type: {canvasCfg['fieldType']}")
            print(f"Debug (Max, Min): ({maxScale}, {minScale}) colour: ({stoColours[-1]}, {stoColours[colorScaleIndex[0][0]]}) Type: {canvasCfg['fieldType']}")

            # All drawing is done at the bottom of the node
            def horCoreBoundary(tuple_in):
                if tuple_in[0] == jsonObject.rebuiltModel.yIndexesYoke[0] and tuple_in[1] in jsonObject.rebuiltModel.toothArray + jsonObject.rebuiltModel.slotArray:
                    return True
                elif tuple_in[0] == jsonObject.rebuiltModel.yIndexesUpperSlot[0] and tuple_in[1] in jsonObject.rebuiltModel.slotArray:
                    return True
                elif tuple_in[0] == jsonObject.rebuiltModel.yIndexesMEC[-1] + 1 and tuple_in[1] in jsonObject.rebuiltModel.toothArray:
                    return True
                else:
                    return False

            # All drawing is done to the right of the node
            def vertCoreBoundary(tuple_in):
                if tuple_in[0] in jsonObject.rebuiltModel.yIndexesMEC:

                    # Left end tooth
                    if tuple_in[1] in [jsonObject.rebuiltModel.toothArray[0] - 1] + [jsonObject.rebuiltModel.toothArray[jsonObject.rebuiltModel.ppEndTooth - 1]]:
                        return True
                    # Right end tooth
                    elif tuple_in[1] in [jsonObject.rebuiltModel.toothArray[-jsonObject.rebuiltModel.ppEndTooth-1]] + [jsonObject.rebuiltModel.toothArray[-1]]:
                        return True
                    # Remaining teeth left edge
                    elif tuple_in[0] not in jsonObject.rebuiltModel.yIndexesYoke and tuple_in[1] in jsonObject.rebuiltModel.toothArray[jsonObject.rebuiltModel.ppEndTooth:-jsonObject.rebuiltModel.ppEndTooth:jsonObject.rebuiltModel.ppSlotpitch]:
                        return True
                    # Remaining teeth right edge
                    elif tuple_in[0] not in jsonObject.rebuiltModel.yIndexesYoke and tuple_in[1] in jsonObject.rebuiltModel.toothArray[jsonObject.rebuiltModel.ppEndTooth + jsonObject.rebuiltModel.ppTooth:-jsonObject.rebuiltModel.ppEndTooth:jsonObject.rebuiltModel.ppSlotpitch]:
                        return True
                    else:
                        return False

                else:
                    return False

            nodeIndexes = [(node.yIndex, node.xIndex) for row in jsonObject.rebuiltModel.matrix for node in row]
            horCoreBoundaryIdxs = filter(horCoreBoundary, nodeIndexes)
            vertCoreBoundaryIdxs = filter(vertCoreBoundary, nodeIndexes)

            xEndTeethBounds = [jsonObject.rebuiltModel.toothArray[0],
                               jsonObject.rebuiltModel.toothArray[-1],
                               jsonObject.rebuiltModel.slotArray[jsonObject.rebuiltModel.m * jsonObject.rebuiltModel.ppSlot - 1] + 1]

            # Assigns a colour to a node based on its relative position in the range of values and the range of available colours
            i, j, k = 0, 0, 0
            while i < jsonObject.rebuiltModel.matrix.shape[0]:
                while j < jsonObject.rebuiltModel.matrix.shape[1]:
                    if canvasCfg["showFilter"]:
                        if i in filteredRows or (i, j) in filteredRowCols:
                            overRideColour = determineColour(jsonObject, i, j,
                                                             [canvasCfg["fieldType"], fieldsScale, stoColours, cPosInf, cNegInf],
                                                             highlightZeroValsInField=canvasCfg["showZeros"])

                        else:
                            overRideColour = '#000000'
                    else:
                        overRideColour = determineColour(jsonObject, i, j,
                                                         [canvasCfg["fieldType"], fieldsScale, stoColours, cPosInf, cNegInf],
                                                         highlightZeroValsInField=canvasCfg["showZeros"])

                    jsonObject.rebuiltModel.matrix[i, j].drawNode(canvasSpacing=jsonObject.rebuiltModel.Cspacing,
                                                                  overRideColour=overRideColour, c=cFields.canvas,
                                                                  nodeWidth=1,
                                                                  outline='blue' if j in xEndTeethBounds else 'black')
                    j += 1
                    k += 1
                j = 0
                i += 1

        cFields.canvas.scale("all", 0, 0, 1, invertCoeff)
        cFields.canvas.configure(scrollregion=cFields.canvas.bbox("all"))
        cFields.root.mainloop()

    if canvasCfg["showMatrix"]:
        visualizeMatrix(dims, jsonObject.rebuiltModel, ogModel, bShowA=True)
        visualizeMatrix(dims, jsonObject.rebuiltModel, ogModel, bShowB=True)

