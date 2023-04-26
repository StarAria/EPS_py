# ---------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------
# --Project: EPS
# --Date: 2023.04.23
# --
# --File: predictor.py
# --Note: Expansion point predictor using RandomForest algorythm.
# --Designers: Wang Chuyu
# --Writers: Wang Chuyu
# ---------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------

import numpy as np

# constant

RESULT_FILE_NAME = "./result.txt"

class Predictor(object):
    '''
    Expansion point predictor using RandomForest algorythm.
    Need data from CaseDatabase.
    Attributes: freqModel, orderModel, freqPedictingResult, orderPedictingResult
    Function: buildModel, predict, showOobError, showPredictingResult, savePredictingResult
    '''

    def __init__(self, db = None):
        '''
        Do nothing if db is not given.
        Train and predict 2nd expanding point order and frequency if db is given. 
        '''
        if(db == None):
            self.freqPedictingResult = []
            self.orderPedictingResult = []
        else:
            self.buildModel()
            self.predict()
        return

    def buildModel(self, item = "both"):
        '''
        Build 2nd expansion point frequency and order predicting model.
        item: "frequency" ---- Build 2nd expansion point frequency predicting model.
              "order" ---- Build 2nd expansion point order predicting model.
              "both" ---- Build both model.
              otherwise ---- Do not build.
        '''
        pass

    def predict(self, featureTable, item = "both", showOobError = False, showResult = True,\
                dumpFile = True):
        '''
        Predict 2nd expansion point frequency and order.
        featureTable: Feature data of cases to be predicted.
        item: "frequency" ---- Build 2nd expansion point frequency predicting model.
              "order" ---- Build 2nd expansion point order predicting model.
              "both" ---- Build both model.
              otherwise ---- Do not build.
        showResult: Print predicting result. True as default.
        showOobError: Plot oob error. False as default.
        dumpFile: Dump predicting result to txt file. True as default.
        '''
        pass

    def showOobError(self, item = "both"):
        '''
        Show out-of-bag error of the model.
        item: "frequency" ---- Plot frequency predicting model oob error.
              "order" ---- Plot order predicting model oob error.
              "both" ---- Plot both model oob error.
              otherwise ---- Do not Plot.
        '''
        pass

    def showPredictingResult(self, item = "both"):
        '''
        Show the predicting result of 2nd expansion point.
        item: "frequency" ---- Print 2nd expansion point frequency predicting result.
              "order" ---- Print 2nd expansion point order predicting model.
              "both" ---- Print both Result.
              otherwise ---- Do not print.
        '''
        pass

    def savePredictingResult(self, item = "both", fileName = RESULT_FILE_NAME):
        '''
        Dump predicting result to txt file.
        item: "frequency" ---- Dump 2nd expansion point frequency predicting result.
              "order" ---- Dump 2nd expansion point order predicting result.
              "both" ---- Dump both result.
              otherwise ---- Do not dump.
        '''
        pass



