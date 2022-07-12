"""IMPORT LIBRARIES"""

import math
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import cmath
import numpy as np
import contextlib
from timeit import default_timer as timer
from collections.abc import MutableMapping
import itertools
import configparser
import pandas as pd
import os

# from numba import cuda, njit, int32, float64
# from numba.experimental import jitclass


PROJECT_PATH = os.path.abspath(os.path.join(__file__, "../.."))
pi = math.pi
j_plex = complex(0, 1)
uo = (10 ** - 7)*4*pi


class LimMotor(object):
    def __init__(self, motorCfg, buildBaseline=False):

        self.errorDict = TransformedDict.buildFromScratch()

        self.copper = Material(1, 8.96, 5.96 * 10 ** 7, 1.72 * 10 ** (-8))
        self.alum = Material(1, 8.96, 17.0 * 10 ** 6, None)
        self.air = Material(1, 8.96, 0, None)
        self.iron = Material(1000, 7.8, 4.5 * 10 ** 6, None)
        self.insul = Material(None, 1.4, None, None)

        # Stator Dimension Variables
        self.m = 3
        self.slots = motorCfg["slots"]
        self.poles = motorCfg["poles"]
        self.q = self.slots/self.poles/self.m
        self.L = motorCfg["length"]  # meters
        self.Tp = self.L/self.poles  # meters

        self.ws = 10 / 1000  # meters
        self.wt = 6 / 1000  # meters
        self.Tper = 0.525  # meters

        self.slotpitch = self.ws + self.wt  # meters
        self.endTooth = self.getLenEndTooth()  # meters
        self.writeErrorToDict(key='name',
                              error=Error.buildFromScratch(name='MotorLength',
                                                           description='ERROR - The inner dimensions do not sum to the motor length',
                                                           cause=not self.validateLength()))

        self.windingLayers = 2
        self.windingShift = motorCfg["windingShift"]
        self.removeUpperCoils = [0] + list(range(self.slots-self.windingShift-1, self.slots-1)) + [self.slots-1]
        self.removeLowerCoils = [0] + list(range(1, 1+self.windingShift)) + [self.slots-1]

        self.Airbuffer = (self.Tper - self.L)/2  # meters
        self.hy = 6.5/1000  # meters
        self.hs = 20/1000  # meters
        self.dr = 2/1000  # meters
        self.g = 2.7/1000  # meters
        self.bi = 8/1000  # meters
        self.D = 50/1000  # meters
        self.H = self.hy + self.hs  # meters
        self.vac = self.g * 1.5

        # Electrical Variables
        self.Ip = np.float64(10)  # AmpsPeak
        self.vel = 0.0  # m/s
        self.f = 100.0  # Hz
        self.t = 0.0  # s
        self.N = 57  # turns

    def writeErrorToDict(self, key, error):
        if error.state:
            self.errorDict.__setitem__(error.__dict__[key], error)

    def getDictKey(self):
        return self.errorDict.keys()

    def getLenEndTooth(self):
        return (self.L - (self.slotpitch * (self.slots - 1) + self.ws)) / 2

    def validateLength(self):
        return round(self.L - (self.slotpitch*(self.slots-1)+self.ws+2*self.endTooth), 12) == 0


class Material(object):
    def __init__(self, ur, density, sigma, resistivity):
        self.ur = ur
        self.density = density  # g/cm^3
        self.sigma = sigma  # Sm^-1
        self.resistivity = resistivity  # OhmMeter


class TransformedDict(MutableMapping):
    """A dictionary that applies an arbitrary key-altering
       function before accessing the keys"""

    def __init__(self, kwargs, buildFromJson=False):

        if buildFromJson:
            self.store = dict()
            for key, error in kwargs.items():
                self.__setitem__(key, Error.buildFromJson(error))
            return

        self.store = dict()
        self.update(dict(kwargs))  # use the free update to set keys

    @classmethod
    def buildFromScratch(cls, **kwargs):
        return cls(kwargs=kwargs)

    @classmethod
    def buildFromJson(cls, jsonObject):
        return cls(kwargs=jsonObject, buildFromJson=True)

    def __eq__(self, otherObject):
        if not isinstance(otherObject, TransformedDict):
            # don't attempt to compare against unrelated types
            return NotImplemented
        # If the objects are the same then set the IDs to be equal
        elif self.__dict__.items() == otherObject.__dict__.items():
            for attr, val in otherObject.__dict__.items():
                return self.__dict__[attr] == otherObject.__dict__[attr]
        # The objects are not the same
        else:
            pass

    def __getitem__(self, key):
        return self.store[key]

    def __setitem__(self, key, value):
        self.store[key] = value

    def __delitem__(self, key):
        del self.store[key]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)

    def genStoreByValueAttr(self, strName):
        return (self.store[strName] for _ in range(self.store.__len__()))

    def printErrorsByAttr(self, attrString):
        cnt = 1
        for key in self.store:
            print(f'{cnt}) {self.__getitem__(key).__dict__[attrString]}')
            cnt += 1

    def isEmpty(self):
        return False if self.store else True


class Error(object):
    def __init__(self, kwargs, buildFromJson=False):

        if buildFromJson:
            for key in kwargs:
                self.__dict__[key] = kwargs[key]
            return

        self.name = kwargs['name']
        self.description = kwargs['description']
        self.cause = bool(kwargs['cause'])  # This was done to handle np.bool_ not being json serializable
        self.state = False

        self.setState()

    @classmethod
    def buildFromScratch(cls, **kwargs):
        return cls(kwargs=kwargs)

    @classmethod
    def buildFromJson(cls, jsonObject):
        return cls(kwargs=jsonObject, buildFromJson=True)

    def __eq__(self, otherObject):
        if not isinstance(otherObject, Error):
            # don't attempt to compare against unrelated types
            return NotImplemented
        # If the objects are the same then set the IDs to be equal
        elif self.__dict__.items() == otherObject.__dict__.items():
            for attr, val in otherObject.__dict__.items():
                return self.__dict__[attr] == otherObject.__dict__[attr]
        # The objects are not the same
        else:
            pass

    def setState(self):
        if self.cause:
            self.state = True
        else:
            self.state = False


def rebuildPlex(val):
    try:
        if val[0] == 'plex_Signature':
            return np.cdouble(val[1] + j_plex * val[2])
    except TypeError:
        return val


def getCurrDensityTable(key):
    """
    Note: The values in the WireSpecs sheet are not accounting for insulation
    Note: All current is in amps
    """
    xlsxFile = os.path.join(PROJECT_PATH, 'SupportingDocs\\Calculators\\CurrentDensityChart.xlsx')
    xl_file = pd.ExcelFile(xlsxFile, engine='openpyxl')

    dfs = {sheet_name: xl_file.parse(sheet_name)
           for sheet_name in xl_file.sheet_names}

    return dfs[key]


currentTable = getCurrDensityTable('WireSpecs')['Current']
diamTable = getCurrDensityTable('WireSpecs')['Dia in mm ']
areaTable = getCurrDensityTable('WireSpecs')['Area in mm.sq']

matList = np.array([('iron', 'gray10'), ('copperA', 'red'), ('copperB', 'green'), ('copperC', 'DodgerBlue2'),
                    ('aluminum', '#717676'), ('vacuum', '#E4EEEE')])


if __name__ == '__main__':
    pass
