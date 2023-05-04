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

import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# constant

RESULT_FILE_NAME = "./result.txt"

FREQ_CRITERION = 'entropy'
ORDER_CRITERION = 'gini'
FREQ_ESTIMATORS = 30
ORDER_ESTIMATORS = 500
FERQ_MAX_DEPTH = 30  
ORDER_MAX_DEPTH = 20
FREQ_MIN_SAMPLES_SPLIT = 2
ORDER_MIN_SAMPLES_SPLIT = 2
FREQ_MIN_SAMPLES_LEAF = 1
OERDER_MIN_SAMPLES_LEAF = 7


class Predictor(object):
    '''
    Expansion point predictor using RandomForest algorythm.
    Need data from CaseDatabase.
    Attributes: _freqModel, _orderModel, freqPredictingResult, orderPredictingResult, tuning
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
            self._freqModel = RandomForestClassifier(oob_score = True)
            self._freqModel.fit(trainingLabel, np.ravel(trainingFrequency))
            print("Out-of bag score: " + str(self._freqModel.oob_score_) + "\n")

        if(item == "order" or item == "both"):
            print("********Build the order predicting model********\n")
            self._orderModel = RandomForestClassifier(oob_score = True)
            self._orderModel.fit(trainingLabel, np.ravel(trainingOrder))
            print("Out-of bag score: " + str(self._orderModel.oob_score_) + "\n")

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
            result = self._freqModel.predict(predictingLabel)
            self.freqPredictingResult = [[featureTable[i, 0], float(result[i])] for i in range(len(result))]
            print("...Done\n")

        if(item == "order" or item == "both"):
            print("********Predict 2nd expansion point order********\n")
            result = self._orderModel.predict(predictingLabel)
            self.orderPredictingResult = [[featureTable[i, 0], int(result[i])] for i in range(len(result))]
            print("...Done\n")

        if(showResult):
            self.showPredictingResult(item=item, merge=True)

        if(dumpFile):
            self.savePredictingResult(item=item, merge=True)

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
            print("Frequency predicting result:")
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

    def tuning(self, trainingDataTable, item, paraName, repetition, paraLst = None):
        '''
        Tune the model.
        '''
        if(item not in ["frequency", "order"]):
            raise ValueError("Unknown parameter \"item=" + item + "\"")
        
        if(type(trainingDataTable) != np.matrix):
            trainingDataTable = np.matrix(trainingDataTable)
        trainingLabel = trainingDataTable[:, 1:-3]

        if(item == "frequency"):
            trainingTarget = trainingDataTable[:, -2]
            myCriterion = FREQ_CRITERION
            myEstimators = FREQ_ESTIMATORS
            myMaxDepth = FERQ_MAX_DEPTH
            myMinSamplesSplit = FREQ_MIN_SAMPLES_SPLIT
            myMinSamplesLeaf = FREQ_MIN_SAMPLES_LEAF
        else:
            trainingTarget = trainingDataTable[:, -3]
            myCriterion = ORDER_CRITERION
            myEstimators = ORDER_ESTIMATORS
            myMaxDepth = ORDER_MAX_DEPTH
            myMinSamplesSplit = ORDER_MIN_SAMPLES_SPLIT
            myMinSamplesLeaf = OERDER_MIN_SAMPLES_LEAF

        if(paraName == 'criterion'):
            print("********Tune the criterion********\n")
            giniScore = []
            entropyScore = []
            for i in range(repetition):
                mdl = RandomForestClassifier(oob_score = True, criterion='gini')
                mdl.fit(trainingLabel, np.ravel(trainingTarget))
                giniScore.append(mdl.oob_score_)
                mdl = RandomForestClassifier(oob_score = True, criterion='entropy')
                mdl.fit(trainingLabel, np.ravel(trainingTarget))
                entropyScore.append(mdl.oob_score_)
            #print(giniScore)
            #print(entropyScore)
            print("gini score: " + str(np.mean(giniScore)))
            print("entropy score: " + str(np.mean(entropyScore)))
            print("...Done\n")

        if(paraName == "n_estimators"):
            print("********Tune the n_estimators********\n")
            score = []
            for i in paraLst:
                buffer = []
                for j in range(repetition):
                    mdl = RandomForestClassifier(oob_score = True, criterion=myCriterion, n_estimators=i)
                    mdl.fit(trainingLabel, np.ravel(trainingTarget))
                    buffer.append(mdl.oob_score_)
                score.append(np.mean(buffer))
            plt.figure(paraName)
            plt.title(item + " model")
            plt.plot(paraLst, score, linewidth=1)
            plt.xlabel(paraName)
            plt.ylabel("score")
            plt.savefig(paraName)
            print("...Done\n")

        if(paraName == "max_depth"):
            print("********Tune the max_depth********\n")
            score = []
            for i in paraLst:
                buffer = []
                for j in range(repetition):
                    mdl = RandomForestClassifier(oob_score = True, criterion=myCriterion, n_estimators=myEstimators,\
                                                 max_depth=i)
                    mdl.fit(trainingLabel, np.ravel(trainingTarget))
                    buffer.append(mdl.oob_score_)
                score.append(np.mean(buffer))
            plt.figure(paraName)
            plt.title(item + " model")
            plt.plot(paraLst, score, linewidth=1)
            plt.xlabel(paraName)
            plt.ylabel("score")
            plt.savefig(paraName)
            print("...Done\n")

        if(paraName == "min_samples_split"):
            print("********Tune the min_samples_split********\n")
            score = []
            for i in paraLst:
                buffer = []
                for j in range(repetition):
                    mdl = RandomForestClassifier(oob_score = True, criterion=myCriterion, n_estimators=myEstimators,\
                                                 max_depth=myMaxDepth, min_samples_split=i)
                    mdl.fit(trainingLabel, np.ravel(trainingTarget))
                    buffer.append(mdl.oob_score_)
                score.append(np.mean(buffer))
            plt.figure(paraName)
            plt.title(item + " model")
            plt.plot(paraLst, score, linewidth=1)
            plt.xlabel(paraName)
            plt.ylabel("score")
            plt.savefig(paraName)
            print("...Done\n")

        if(paraName == "min_samples_leaf"):
            print("********Tune the min_samples_leaf********\n")
            score = []
            for i in paraLst:
                buffer = []
                for j in range(repetition):
                    mdl = RandomForestClassifier(oob_score = True, criterion=myCriterion, n_estimators=myEstimators,\
                                                 max_depth=myMaxDepth, min_samples_split=myMinSamplesSplit,\
                                                 min_samples_leaf=i)
                    mdl.fit(trainingLabel, np.ravel(trainingTarget))
                    buffer.append(mdl.oob_score_)
                score.append(np.mean(buffer))
            plt.figure(paraName)
            plt.title(item + " model")
            plt.plot(paraLst, score, linewidth=1)
            plt.xlabel(paraName)
            plt.ylabel("score")
            plt.savefig(paraName)
            print("...Done\n")
       
        return

    def _tryModel(self, para):
        '''
        Build model from paralst.
        Used for parallel fine tune.
        '''
        [x, y, myCriterion, myEstimators, myMaxDepth, myMinSamplesSplit, myMinSamplesLeaf] = para
        score = []
        for i in range(100):
            mdl = RandomForestClassifier(oob_score = True, criterion=myCriterion, n_estimators=myEstimators,\
                                 max_depth=myMaxDepth, min_samples_split=myMinSamplesSplit,\
                                 min_samples_leaf=myMinSamplesLeaf)
            mdl.fit(x, np.ravel(y))
            score.append(mdl.oob_score_)

        return np.mean(score)
    
    def fineTune(self, item, trainingDataTable,\
                 criterion_lst, n_estimators_lst, max_depth_lst, min_samples_split_lst, min_samples_leaf_lst):
        '''
        Fine tune the model.
        Use grid search method.
        '''

        print("********Fine tune********\n")
        if(item not in ["frequency", "order"]):
            raise ValueError("Unknown parameter \"item=" + item + "\"")
        
        if(type(trainingDataTable) != np.matrix):
            trainingDataTable = np.matrix(trainingDataTable)
        x = trainingDataTable[:, 1:-3]

        if(item == "frequency"):
            y = trainingDataTable[:, -2]
        else:
            y = trainingDataTable[:, -3]

        paras = []
        for i in criterion_lst:
            for j in n_estimators_lst:
                for k in max_depth_lst:
                    for m in min_samples_split_lst:
                        for n in min_samples_leaf_lst:
                            paras.append([x, y, i, j, k, m, n])
        
        # Create a pool of worker processes
        with multiprocessing.Pool(processes=20) as pool:
            scores = pool.map(self._tryModel, paras)

        print("mean score:" + str(np.mean(scores)) + "\n")

        sorted_id = sorted(range(len(scores)), key = lambda k: scores[k], reverse = True)

        print("Top 20 scores:")
        print("\tscore\t\tcriterion\tn_estimators\tmax_depth\tmin_samples_split\tmin_samples_leaf")
        for i in sorted_id[:20]:
            print("\t" + f'{scores[i]:.6f}' + "\t" + str(paras[i][2]) + "\t\t" + str(paras[i][3]) +\
                  "\t\t" + str(paras[i][4]) + "\t\t" + str(paras[i][5]) + "\t\t\t" + str(paras[i][6]))
            
        print("\n...Done\n")
        return
    
    









# ---------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------
# Unit Test

if __name__ == "__main__":
    from case_database import CaseData, CaseDatabase

    db = CaseDatabase(trainingCaseDir="../case/Training", predictingCaseDir="../case/Predicting", build=False)
    # db.loadPredictingFeature()
    db.loadTrainingFeature()
    predictor = Predictor()
    # predictor.tuning(db.trainingFeatureTable(), "order", "min_samples_leaf", 30, range(4, 8, 1))
    # predictor.fineTune("order", db.trainingFeatureTable(),\
    #                    ["gini", 'entropy'], [480, 485, 490, 495, 500], [18, 20, 22],\
    #                     [2, 3, 4,], [6, 7, 8])



