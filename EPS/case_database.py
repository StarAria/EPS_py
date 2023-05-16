# ---------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------
# --Project: EPS
# --Date: 2023.04.17
# --
# --File: case_database.py
# --Note: define the database of varies circuit cases.
# --Designers: Wang Chuyu
# --Writers: Wang Chuyu
# ---------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------

import os
import multiprocessing
import numpy as np
from scipy.sparse import *
from scipy import constants
from scipy.linalg import solve
from scikits.umfpack import spsolve

# constant
PROCESS_CNT = 16
ERROR_MAX = 1e10
ERROR_SAMPLE_COUNT = 400
FREQ_LIMIT = 500e9
SAMPLE_STEP = 20e9
TOTAL_ORDER = 10
TRAINING_CASE_DIR = "../case/Training"
PREDICTING_CASE_DIR = "../case/Predicting"

class CaseData(object):
    '''
    Data for single case
    Attributes: name, order, bCol, c, g, b
    Function: setB, setC, setG, featureExtract, arnoldi, errorEval, measureMinError
    '''

    def __init__(self, name, order, bCol):
        self.name = name
        self.order = order
        self.bCol = bCol

    def setB(self, b):
        '''
        Set B matrix in CaseData.
        The type of B will be coo_matrix
        '''
        
        if(b.shape != (self.order, self.bCol)):
            print("********Error: shape mismatch in B matrix of case " + self.name + " ********\n")
            print("********" + str((self.order, self.bCol)) + " needed, but " + str(b.shape) + " is given********\n")
            print("********skipping setting B matrix of case " + self.name + " ********\n")
            return
        
        self.b = coo_matrix(b)
        return
    
    def setC(self, c):
        '''
        Set C matrix in CaseData.
        The type of C will be coo_matrix
        '''
        
        if(c.shape != (self.order, self.order)):
            print("********Error: shape mismatch in C matrix of case " + self.name + " ********\n")
            print("********" + str((self.order, self.order)) + " needed, but " + str(c.shape) + " is given********\n")
            print("********skipping setting C matrix of case " + self.name + " ********\n")
            return
        
        self.c = coo_matrix(c)
        return
    
    def setG(self, g):
        '''
        Set G matrix in CaseData.
        The type of G will be coo_matrix
        '''
        
        if(g.shape != (self.order, self.order)):
            print("********Error: shape mismatch in G matrix of case " + self.name + " ********\n")
            print("********" + str((self.order, self.order)) + " needed, but " + str(g.shape) + " is given********\n")
            print("********skipping setting G matrix of case " + self.name + " ********\n")
            return
        
        self.g = coo_matrix(g)
        return
    
    def arnoldi(self, k, s):
        '''
        Arnoldi method to get the orthonormal matrix V
        May throw error if the matrix d = g + s * c is singular
        k: reduction order
        s: expansion point frequency
        '''
        c = self.c.todense()
        g = self.g.todense()
        b = self.b.todense()

        v = np.matrix(np.zeros((self.order, k)))

        d = csr_matrix(g + s * c) 
        # s = splu(csc_matrix(d))
        r = spsolve(d, b)
        r = np.matrix(r / np.linalg.norm(r)).T
        v[:, 0] = r

        for i in range(1, k):
            # z = s.solve(c * v[:, i - 1])
            z = spsolve(d, c * v[:, i - 1])
            z = np.matrix(z).T
            h = v.T * z
            z = z - v * h
            h = v.T * z
            z = z - v * h
            v[:, i] = z / np.linalg.norm(z)

        return v
    
    def featureExtract(self):
        '''
        Extract the case features.
        cMax, cMin, gMax, gMin: the max and min element in C and G matrix.
        cMean, gMean: the mean value of C and G matrix.
        cColVar, gColVar: the varience of column mean value of C and G matrix.
        cNnz, gNnz: the none zero element count in C and G matrix.
        cgElemProduct: the mean value of the product of each corresponding element in C and G matrix.
        dim: the dimension of the C and G matrix.
        '''
        cMax = self.c.max()
        gMax = self.g.max()
        cMin = self.c.min()
        gMin = self.g.min()
        cMean = self.c.mean()
        gMean = self.g.mean()
        cColVar = np.var(self.c.mean(axis = 0))
        gColVar = np.var(self.g.mean(axis = 0))
        cNnz = self.c.nnz
        gNnz = self.g.nnz
        cgElemProduct = self.c.multiply(self.g).mean()
        dim = self.order
        return [self.name, cMax, gMax, cMin, gMin, cMean, gMean, cColVar, gColVar,\
                cNnz, gNnz, cgElemProduct, dim]
    
    def errorEval(self, v, fLim, sampleCnt):
        '''
        Evaluate order reduction error with specific tranformation matrix V.
        May throw error when the matrix to be solved is singulr.
        v: tranformation matrix given by Arnoldi method.
        fLim: measurement frenquency upper limit.
        sampleCnt: total measurement sample point count.
        '''
        rb = v.T * self.b
        rc = v.T * self.c.todense() * v
        rg = v.T * self.g.todense() * v

        x = np.matrix(np.zeros((self.order, sampleCnt + 1)), dtype=complex)
        rx = np.matrix(np.zeros((self.order, sampleCnt + 1)), dtype=complex)
        step = fLim / sampleCnt

        for i in range(0, sampleCnt + 1):
            f = i * step
            s = 2j * constants.pi * f
            # originSolver = splu(csc_matrix(self.g + s * self.c))
            tmp = spsolve(csc_matrix(self.g + s * self.c), self.b.todense())
            x[:, i] = np.matrix(tmp).T
            t = solve((rg + s * rc), rb)
            rx[:, i] = v * t

        meanError = (abs(x - rx) / abs(x)).mean()
        return meanError
            
    def measureMinError(self, totalOrder = TOTAL_ORDER, maxFreq = FREQ_LIMIT, step = SAMPLE_STEP,\
                        sampleCnt = ERROR_SAMPLE_COUNT):
        '''
        Find the (frequence, order) of the 2nd expasion point
        to minimum the total error of the 1st and 2nd expansion point.
        The 1st expansion point is always at 0Hz.
        totalOrder: the sum of the order after reducion on point 1 and 2.
        maxFreq: maximum frequency of the 2nd expansion point.
        step: sample step of the 2nd expansion point frequency.
        '''
        feature = self.featureExtract()
        error = ERROR_MAX

        for i in range(1, totalOrder):
            k1 = totalOrder - i
            k2 = i
            j = step
            while j <= maxFreq:
                s1 = 0
                s2 = j
                try:
                    v1 = self.arnoldi(k1, s1) 
                    v2 = self.arnoldi(k2, s2)
                    error1 = self.errorEval(v1, maxFreq, sampleCnt)
                    error2 = self.errorEval(v2, maxFreq, sampleCnt)
                except(np.linalg.LinAlgError):
                    j = j + step
                    continue
                if error1 + error2 <= error:
                    error = error1 + error2
                    result = feature + [i, j, error]
                j = j + step

        print("case " + self.name + " done.")
        return result


class CaseDatabase(object):
    '''
    Database for cases. Training and predicting cases are included.
    Apply access and parser to outer files.
    Attributes: _trainingCaseList, _predictingCaseList, trainingFeatureTable, predictingFeatureTable
    Function: parser, buildCaseData, measureTrainingData, extractPredictingFeature, trainingList, predictingList, 
    saveTrainingFeature, savePredictingFeature, loadTrainingFeature, loadPredictingFeature
    '''

    def __init__(self, trainingCaseDir = TRAINING_CASE_DIR, predictingCaseDir = PREDICTING_CASE_DIR, build = True):
        '''
        if build = True:
        Read files from case directory.
        Build _trainingCaseList and _predictingCaseList using buildCaseData function.
        '''
        self.trainingCaseDir = trainingCaseDir
        self.predictingCaseDir = predictingCaseDir
        if build:
            self.buildCaseData()
            self.measureTrainingData()
            self.extractPredictingFeature()
            self.savePredictingFeature()
            self.saveTrainingFeature()

        else:
            self._predictingCaseList = []
            self._trainingCaseList = []
            self._predictingFeatureTable = []
            self._trainingFeatureTable = []

        return

    def parser(self, fileName):
        '''
        Read and build sparse matrix from file.
        Matrix files in COO format are supported.
        Do not construct matrix since size is unknown.
        '''
        # read the file and extract the data
        data = np.loadtxt(fileName)
        if data.ndim == 1:  # handle the case when there is only one line in the file
            rows = [int(data[0])]
            cols = [int(data[1])]
            values = [data[2]]
        else:
            rows = data[:, 0].astype(int)
            cols = data[:, 1].astype(int)
            values = data[:, 2]
        # return the result
        return [rows, cols, values]


    def trainingFeatureTable(self):
        '''
        Access to _trainingFeatureTable.
        '''
        return self._trainingFeatureTable
    
    def predictingFeatureTable(self):
        '''
        Access to _predictingFeatureTable.
        '''
        return self._predictingFeatureTable

    def buildCaseData(self, dataSet = "both", trainingDir = None, predictingDir = None):
        '''
        Load data from case directory.
        Rebuild _trainingCaseList and/or _predictingCaseList.
        This method do not rebuild _trainingFeatureTable and _predictingFeatureTable.
        dataSet: "training" ---- Rebuild _trainingCaseList from trainingDir.
                 "predicting" ---- Rebuild _predictingCaseList from predictingDir.
                 "both" ---- Rebuild both dir.
                 otherwise ---- Do not rebuild.
        trainingDir: training case directory. self.trainingCaseDir as default.
        predictingDir: predicting case directory. self.predictingCaseDir as default.
        '''
        if(dataSet not in ["training", "predicting", "both"]):
            raise ValueError("Unknown parameter \"dataset=" + dataSet + "\"")
        
        if(trainingDir == None):
            trainingDir = self.trainingCaseDir

        if(predictingDir == None):
            predictingDir = self.predictingCaseDir

        if(dataSet == "both" or dataSet == "training"):
            print("********Begin to build trainingCaseList********\n")
            self._trainingCaseList = []
            caseNameList = os.listdir(trainingDir)
            for fileName in caseNameList:
                infoFileName = trainingDir + "/" + fileName + "/" + fileName + ".txt"
                bFileName = trainingDir + "/" + fileName + "/" + fileName + "_B.txt"
                cFileName = trainingDir + "/" + fileName + "/" + fileName + "_C.txt"
                gFileName = trainingDir + "/" + fileName + "/" + fileName + "_G.txt"
                with open(infoFileName, 'r') as infoF:
                    info = infoF.readlines()
                    case = CaseData(str(info[0].split()[0]), int(info[1].split()[0]), int(info[2].split()[0]))
                [rows, cols, values] = self.parser(bFileName)
                case.setB(coo_matrix((values, (rows, cols)), shape=(case.order, case.bCol)))
                [rows, cols, values] = self.parser(cFileName)
                case.setC(coo_matrix((values, (rows, cols)), shape=(case.order, case.order)))
                [rows, cols, values] = self.parser(gFileName)
                case.setG(coo_matrix((values, (rows, cols)), shape=(case.order, case.order)))
                self._trainingCaseList.append(case)
            print("Build trainingCaseList done, " + str(len(self._trainingCaseList))\
                   + " cases added to list\n")

        if(dataSet == "both" or dataSet == "predicting"):
            print("********Begin to build predictingCaseList********\n")
            self._predictingCaseList = []
            caseNameList = os.listdir(predictingDir)
            for fileName in caseNameList:
                infoFileName = predictingDir + "/" + fileName + "/" + fileName + ".txt"
                bFileName = predictingDir + "/" + fileName + "/" + fileName + "_B.txt"
                cFileName = predictingDir + "/" + fileName + "/" + fileName + "_C.txt"
                gFileName = predictingDir + "/" + fileName + "/" + fileName + "_G.txt"
                with open(infoFileName, 'r') as infoF:
                    info = infoF.readlines()
                    case = CaseData(str(info[0].split()[0]), int(info[1].split()[0]), int(info[2].split()[0]))
                [rows, cols, values] = self.parser(bFileName)
                case.setB(coo_matrix((values, (rows, cols)), shape=(case.order, case.bCol)))
                [rows, cols, values] = self.parser(cFileName)
                case.setC(coo_matrix((values, (rows, cols)), shape=(case.order, case.order)))
                [rows, cols, values] = self.parser(gFileName)
                case.setG(coo_matrix((values, (rows, cols)), shape=(case.order, case.order)))
                self._predictingCaseList.append(case)
            print("Build predictingCaseList done, " + str(len(self._predictingCaseList))\
                   + " cases added to list\n")
            
        return

    def _measureMinError(self, case):
        '''
        Case min error measurement for parallel execution.
        '''
        return case.measureMinError()

    def measureTrainingData(self, processCnt = PROCESS_CNT):
        '''
        Build _trainingFeatureTable from _trainingCaseList.
        This method will delete the existing _trainingFeatureTable.
        May be time-consuming.
        '''
        print("********Begin to measure training data********\n")
        print("This may take tens of minutes...\n")

        self._trainingFeatureTable = []
        # Create a pool of worker processes
        with multiprocessing.Pool(processes=processCnt) as pool:
        # Apply the feature extract function to each training case in parallel
            self._trainingFeatureTable = pool.map(self._measureMinError, self._trainingCaseList)

        print("...Done\n")
        return

    def extractPredictingFeature(self):
        '''
        Build _predictingFeatureTable from _predictingCaseList.
        This method will delete the existing _predictingFeatureTable.
        '''
        print("********Begin to extract predicting data feature********\n")

        self._predictingFeatureTable = []
        for case in self._predictingCaseList:
            self._predictingFeatureTable.append(case.featureExtract())

        print("...Done\n")
        return

    def saveTrainingFeature(self, fileName = "./training_feature.txt"):
        '''
        Save _trainingFeatureTable to specific file.
        '''
        print("********Save training feature table********\n")
        print("Target: " + fileName)
        with open(fileName, 'w') as f:
            for case in self._trainingFeatureTable:
                line = '\t'.join(map(str, case)) + '\n'
                f.write(line)
        print("...Done\n")
        return

    def savePredictingFeature(self, fileName = "./predicting_feature.txt"):
        '''
        Save _predictingFeatureTable to specific file.
        '''
        print("********Save predicting feature table********\n")
        print("Target: " + fileName)
        with open(fileName, 'w') as f:
            for case in self._predictingFeatureTable:
                line = '\t'.join(map(str, case)) + '\n'
                f.write(line)
        print("...Done\n")
        return

    def loadTrainingFeature(self, fileName = "./training_feature.txt"):
        '''
        Load _trainingFeatureTable from specific file.
        '''
        print("********Load training feature table********\n")
        print("Target: " + fileName)
        table = []
        with open(fileName, 'r') as f:
            for line in f:
                case = line.strip().split('\t')
                case = [case[0]] + list(map(float, case[1:9])) + list(map(int, case[9:11])) +\
                      [float(case[11])] + [int(case[12])] +\
                      [int(case[13])] + [float(case[14])] + [float(case[15])]
                table.append(case)
            self._trainingFeatureTable = table
        print("...Done\n")
        return


    def loadPredictingFeature(self, fileName = "./predicting_feature.txt"):
        '''
        Load _PredictingFeatureTable from specific file.
        '''
        print("********Load predicting feature table********\n")
        print("Target: " + fileName)
        table = []
        with open(fileName, 'r') as f:
            for line in f:
                case = line.strip().split('\t')
                case = [case[0]] + list(map(float, case[1:9])) + list(map(int, case[9:11])) +\
                      [float(case[11])] + [int(case[12])]
                table.append(case)

            self._predictingFeatureTable = table
        print("...Done\n")
        return
    
    def measureAllCases(self, frequency, order, totalOrder = TOTAL_ORDER, maxFreq = FREQ_LIMIT,\
                     step = SAMPLE_STEP, sampleCnt = ERROR_SAMPLE_COUNT, dataSet = "both"):
        '''
        Measure all cases' error with 2nd expansion point(frequency, order)
        frequency: 2nd expansion point frequency
        order: 2nd expansion point order
        dataSet: "training" ---- Measure cases in _trainingCaseList.
                 "predicting" ---- Measure cases in _predictingCaseList.
                 "both" ---- Measure both dir cases.
                 otherwise ---- Do not rebuild.
        '''
        print("********Begin to measure cases with specific 2nd expansion point********\n")

        if(dataSet not in ["training", "predicting", "both"]):
            raise ValueError("Unknown parameter \"dataset=" + dataSet + "\"")
        
        measureData = []
        if(dataSet == "both" or dataSet == "training"):
            for case in self._trainingCaseList:
                k1 = totalOrder - order
                k2 = order
                s1 = 0
                s2 = frequency
                #try:
                #    v1 = case.arnoldi(k1, s1) 
                #    v2 = case.arnoldi(k2, s2)
                #    error1 = case.errorEval(v1, maxFreq, sampleCnt)
                #    error2 = case.errorEval(v2, maxFreq, sampleCnt)
                #except(np.linalg.LinAlgError):
                #    error = ERROR_MAX
                v1 = case.arnoldi(k1, s1) 
                v2 = case.arnoldi(k2, s2)
                error1 = case.errorEval(v1, maxFreq, sampleCnt)
                error2 = case.errorEval(v2, maxFreq, sampleCnt)
                measureData.append([case.name, error1+error2])
                print("case " + case.name + " done.")

        if(dataSet == "both" or dataSet == "predicting"):
            for case in self._predictingCaseList:
                k1 = totalOrder - order
                k2 = order
                s1 = 0
                s2 = frequency
                #try:
                #    v1 = case.arnoldi(k1, s1) 
                #    v2 = case.arnoldi(k2, s2)
                #    error1 = case.errorEval(v1, maxFreq, sampleCnt)
                #    error2 = case.errorEval(v2, maxFreq, sampleCnt)
                #except(np.linalg.LinAlgError):
                #    error = ERROR_MAX
                v1 = case.arnoldi(k1, s1) 
                v2 = case.arnoldi(k2, s2)
                error1 = case.errorEval(v1, maxFreq, sampleCnt)
                error2 = case.errorEval(v2, maxFreq, sampleCnt)
                measureData.append([case.name, error1+error2])
                print("case " + case.name + " done.")
        
        print("...All done\n")
        return measureData
    
    def _findAndMeasure(self, lst):
        '''
        find and measure single case.
        Use for parallel running measureCase.
        '''
        [trainingNameLst, predictingNameLst, trainingCaseList, predictingCaseList,\
         point, totalOrder, maxFreq, step, sampleCnt] = lst
        [name, freq, order] = point
        ti = -1
        pi = -1
        try:
            ti = trainingNameLst.index(name)
        except:
            pass
        try:
            pi = predictingNameLst.index(name)
        except:
            pass
        if not (ti == -1):
            case = trainingCaseList[ti]
        elif not (pi == -1):
            case = predictingCaseList[pi]
        else:
            raise ValueError("Case " + name + " not found in database.")
        k1 = totalOrder - order
        k2 = order
        s1 = 0
        s2 = freq
        #try:
        #    v1 = case.arnoldi(k1, s1) 
        #    v2 = case.arnoldi(k2, s2)
        #    error1 = case.errorEval(v1, maxFreq, sampleCnt)
        #    error2 = case.errorEval(v2, maxFreq, sampleCnt)
        #except(np.linalg.LinAlgError):
        #    error = ERROR_MAX
        v1 = case.arnoldi(k1, s1) 
        v2 = case.arnoldi(k2, s2)
        error1 = case.errorEval(v1, maxFreq, sampleCnt)
        error2 = case.errorEval(v2, maxFreq, sampleCnt)
        print("case " + name + " done.")
        return([name, error1 + error2])
        

    
    def measureCases(self, expansionPoints, totalOrder = TOTAL_ORDER, maxFreq = FREQ_LIMIT,\
                     step = SAMPLE_STEP, sampleCnt = ERROR_SAMPLE_COUNT, processCnt = PROCESS_CNT):
        '''
        Measure the cases with given expansion points.
        expansionPoints: 2nd expansion point list. Model name, frequency and order are included.
        '''
        print("********Measure error with given case********")

        trainingNameLst = [row[0] for row in self._trainingFeatureTable]
        predictingNameLst = [row[0] for row in self._predictingFeatureTable]
        input = [[trainingNameLst, predictingNameLst, self._trainingCaseList, self._predictingCaseList,\
                  expansionPoints[i], totalOrder, maxFreq, step, sampleCnt]\
                 for i in range(len(expansionPoints))]
        
        with multiprocessing.Pool(processes=processCnt) as pool:
            result = pool.map(self._findAndMeasure, input)
        
        print("...Done\n")
        return result







# ---------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------
# Unit Test

if __name__ == "__main__":

    db = CaseDatabase(False)
    # t = CaseData("414", 10, 1)
    # [rows, cols, values] = db.parser(PREDICTING_CASE_DIR + "/414/414_B.txt")
    # t.setB(coo_matrix((values, (rows, cols)), shape=(10, 1)))
    # [rows, cols, values] = db.parser(PREDICTING_CASE_DIR + "/414/414_C.txt")
    # t.setC(coo_matrix((values, (rows, cols)), shape=(10, 10)))
    # [rows, cols, values] = db.parser(PREDICTING_CASE_DIR + "/414/414_G.txt")
    # t.setG(coo_matrix((values, (rows, cols)), shape=(10, 10)))

    # print(t.b)
    # print(t.c)
    # print(t.g)

    # b = coo_matrix(([-1], ([0], [0])), shape=(10,1))
    # c = coo_matrix(([1.000000e-18, 2.388640e-17, 2.388640e-17, 1.319750e-16, 1.145700e-17, 1.145700e-17],\
    #                 ([1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6])),\
    #                 shape=(10,10))
    # g = np.matrix([[6.171022, 0, 0, 0, -5.080113e+00, 0, 0, 0, 0, -9.090909e-02],\
    #               [0, 5.171022, -9.090909e-02, 0, 0, 0, 0, -5.080113e+00, 0, 0],\
    #               [0, -9.090909e-02, 8.849164, 0, 0, 0, 0, 0, -8.758255e+00, 0],\
    #               [0, 0, 0, 8.767346e+00, 0, 0, -9.090909e-03, 0, 0, -8.758255e+00],\
    #               [-5.080113, 0, 0, 0, 5.186578e+00, 0, 0, -1.064646e-01, 0, 0],\
    #               [0, 0, 0, 0, 0, 9.090909e-03, 0, 0, -9.090909e-03, 0],\
    #               [0, 0, 0, -9.090909e-03, 0, 0, 9.090909e-03, 0, 0, 0],\
    #               [0, -5.080113, 0, 0, -1.064646e-01, 0, 0, 5.186578e+00, 0, 0],\
    #               [0, 0, -8.758255, 0, 0, -9.090909e-03, 0, 0, 8.767346e+00, 0],\
    #               [-9.090909e-02, 0, 0, -8.758255e+00, 0, 0, 0, 0, 0, 8.849164e+00]],)
    
    # v1 = t.arnoldi(4, 0)
    # v2 = t.arnoldi(11, 5e9)

    # print(v1)

    # print(t.errorEval(v1, FREQ_LIMIT, ERROR_SAMPLE_COUNT))
    # print(t.errorEval(v2, FREQ_LIMIT, ERROR_SAMPLE_COUNT))
    # print(t.errorEval(v1, FREQ_LIMIT, ERROR_SAMPLE_COUNT) + t.errorEval(v2, FREQ_LIMIT, ERROR_SAMPLE_COUNT))

    # print(t.measureMinError(15, FREQ_LIMIT, 1e9))

    print(len(db.trainingFeatureTable()))
    print(len(db.predictingFeatureTable()))

    db.loadPredictingFeature()
    db.loadTrainingFeature()
    # db.buildCaseData()
    # print(len(db.trainingFeatureTable()))
    # print(len(db.predictingFeatureTable()))
    # db.measureTrainingData()
    print(len(db.trainingFeatureTable()))
    print(db.trainingFeatureTable()[0])
    print(db.trainingFeatureTable()[399])

    # db.extractPredictingFeature()
    print(len(db.predictingFeatureTable()))
    print(db.predictingFeatureTable()[0])
    print(db.predictingFeatureTable()[99])

    # db.savePredictingFeature()
    # db.saveTrainingFeature()



    











