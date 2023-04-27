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
from sklearn.ensemble import RandomForestClassifier

# constant

RESULT_FILE_NAME = "./result.txt"

class Predictor(object):
    '''
    Expansion point predictor using RandomForest algorythm.
    Need data from CaseDatabase.
    Attributes: freqModel, orderModel, freqPredictingResult, orderPredictingResult
    Function: buildModel, predict, showOobError, showPredictingResult, savePredictingResult
    '''

    def __init__(self, db = None):
        '''
        Do nothing if db is not given.
        Train and predict 2nd expanding point order and frequency if db is given. 
        '''
        if(db == None):
            self.freqPredictingResult = []
            self.orderPredictingResult = []
        else:
            self.buildModel(db.trainingFeatureTable())
            self.predict(db.predictingFeatureTable())
        return

    def buildModel(self, trainingDataTable, item = "both"):
        '''
        Build 2nd expansion point frequency and order predicting model.
        trainingDataTable: Data of cases to train the model.
        item: "frequency" ---- Build 2nd expansion point frequency predicting model.
              "order" ---- Build 2nd expansion point order predicting model.
              "both" ---- Build both model.
              otherwise ---- Do not build.
        '''
        if(item not in ["frequency", "order", "both"]):
            raise ValueError("Unknown parameter \"item=" + item + "\"")
        
        if(type(trainingDataTable) != np.matrix):
            trainingDataTable = np.matrix(trainingDataTable)
        trainingLabel = trainingDataTable[:, 1:-3]
        trainingFrequency = trainingDataTable[:, -2]
        trainingOrder = trainingDataTable[:, -3]

        if(item == "frequency" or item == "both"):
            print("********Build the frequency predicting model********\n")
            self.freqModel = RandomForestClassifier(oob_score = True)
            self.freqModel.fit(trainingLabel, np.ravel(trainingFrequency))
            print("Out-of bag score: " + str(self.freqModel.oob_score_) + "\n")

        if(item == "order" or item == "both"):
            print("********Build the order predicting model********\n")
            self.orderModel = RandomForestClassifier(oob_score = True)
            self.orderModel.fit(trainingLabel, np.ravel(trainingOrder))
            print("Out-of bag score: " + str(self.orderModel.oob_score_) + "\n")

        return

    def predict(self, featureTable, item = "both", showResult = True,\
                dumpFile = True):
        '''
        Predict 2nd expansion point frequency and order.
        featureTable: Feature data of cases to be predicted.
        item: "frequency" ---- Build 2nd expansion point frequency predicting model.
              "order" ---- Build 2nd expansion point order predicting model.
              "both" ---- Build both model.
              otherwise ---- Do not build.
        showResult: Print predicting result. True as default.
        dumpFile: Dump predicting result to txt file. True as default.
        '''
        if(item not in ["frequency", "order", "both"]):
            raise ValueError("Unknown parameter \"item=" + item + "\"")

        if(type(featureTable) != np.matrix):
            featureTable = np.matrix(featureTable)
        predictingLabel = featureTable[:, 1:]

        if(item == "frequency" or item == "both"):
            print("********Predict 2nd expansion point frequency********\n")
            result = np.matrix(self.freqModel.predict(predictingLabel).astype(np.float)).T
            self.freqPredictingResult = np.concatenate((featureTable[:, 0], result), axis = 1).tolist()
            print("...Done\n")

        if(item == "order" or item == "both"):
            print("********Predict 2nd expansion point order********\n")
            result = np.matrix(self.orderModel.predict(predictingLabel).astype(np.int)).T
            self.orderPredictingResult = np.concatenate((featureTable[:, 0], result), axis = 1).tolist()
            print("...Done\n")

        if(showResult):
            self.showPredictingResult(item=item, merge=False)

        if(dumpFile):
            self.savePredictingResult(item=item, merge=False)

        return

    def showPredictingResult(self, item = "both", merge = False):
        '''
        Show the predicting result of 2nd expansion point.
        item: "frequency" ---- Print 2nd expansion point frequency predicting result.
              "order" ---- Print 2nd expansion point order predicting model.
              "both" ---- Print both Result.
              otherwise ---- Do not print.
        merge: Merge the frequency and order predict result table. 
               Ensure that the two result are of the same cases before set merge = True.
               valid only if item = True.
        '''
        if(item not in ["frequency", "order", "both"]):
            raise ValueError("Unknown parameter \"item=" + item + "\"")

        print("********Predicting Result********\n")
        if ((item == "frequency") or (item == "both" and not merge)):
            print("\nFrequency predicting result:")
            print("\tCase\t\tFrequency")
            for case in self.freqPredictingResult:
                print("\t" + case[0] + "\t\t" + str(case[1]))
            print("\n")

        if ((item == "order") or (item == "both" and not merge)):
            print("Order predicting result:")
            print("\tCase\t\tOrder")
            for case in self.orderPredictingResult:
                print("\t" + case[0] + "\t\t" + str(case[1]))
            print("\n")

        if (item == "both" and merge):
            print("Predicting result:")
            print("\tCase\t\tOrder\t\tFrequency")
            for i in range(len(self.orderPredictingResult)):
                print("\t" +  self.orderPredictingResult[i][0] + "\t\t" + str(self.orderPredictingResult[i][1])\
                      + "\t\t" + str(self.freqPredictingResult[i][1]))
            print("\n")
                
        return
        

    def savePredictingResult(self, item = "both", merge = False, fileName = RESULT_FILE_NAME):
        '''
        Dump predicting result to txt file.
        item: "frequency" ---- Dump 2nd expansion point frequency predicting result.
              "order" ---- Dump 2nd expansion point order predicting result.
              "both" ---- Dump both result.
              otherwise ---- Do not dump.
        merge: Merge the frequency and order predict result table. 
               Ensure that the two result are of the same cases before set merge = True.
               valid only if item = True.        
        '''
        if(item not in ["frequency", "order", "both"]):
            raise ValueError("Unknown parameter \"item=" + item + "\"")

        if ((item == "frequency") or (item == "both" and not merge)):
            dotIndex = fileName.rfind('.')
            if(dotIndex == -1):
                realFileName = fileName + "_frequency"
            else:
                realFileName = fileName[:dotIndex] + "_frequency" + fileName[dotIndex:]
            print("********Dump frequency predicting result********\n")
            print("Target: " + realFileName)
            with open(realFileName, 'w') as f:
                f.write("Case  Frequency\n")
                for case in self.freqPredictingResult:
                    line = '  '.join(map(str, case)) + '\n'
                    f.write(line)
            print("...Done\n")

        if ((item == "order") or (item == "both" and not merge)):
            dotIndex = fileName.rfind('.')
            if(dotIndex == -1):
                realFileName = fileName + "_order"
            else:
                realFileName = fileName[:dotIndex] + "_order" + fileName[dotIndex:]
            print("********Dump order predicting result********\n")
            print("Target: " + realFileName)
            with open(realFileName, 'w') as f:
                f.write("Case  Order\n")
                for case in self.orderPredictingResult:
                    line = '  '.join(map(str, case)) + '\n'
                    f.write(line)
            print("...Done\n")

        if (item == "both" and merge):
            print("********Dump predicting result********\n")
            print("Target: " + fileName)
            with open(fileName, 'w') as f:
                f.write("Case  Order  Frequency\n")
                for i in range(len(self.freqPredictingResult)):
                    f.write(self.orderPredictingResult[i][0] + "  " + str(self.orderPredictingResult[i][1])\
                          + "  " + str(self.freqPredictingResult[i][1]) + "\n")
            print("...Done\n")

        return








# ---------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------
# Unit Test

if __name__ == "__main__":
    from case_database import CaseData, CaseDatabase

    db = CaseDatabase(False)
    db.loadPredictingFeature()
    db.loadTrainingFeature()
    predictor = Predictor(db)



