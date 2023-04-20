# ---------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------
# --Project: EPS
# --Date: 2023.04.17
# --
# --File: case_database.py
# --Note: define the database of varies circuit cases.
# --Designers: Wang Chuyu
# --Writers: Wang CHuyu
# ---------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------

import numpy as np
from scipy.sparse import *
from scipy import constants
from scipy.linalg import solve
from scipy.sparse.linalg import splu

# constant
ERROR_MAX = 1e10
ERROR_SAMPLE_COUNT = 500
FREQ_LIMIT = 2e10
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

        d = g + s * c
        s = splu(csc_matrix(d))
        r = s.solve(b)
        r = r / np.linalg.norm(r)
        v[:, 0] = r

        for i in range(1, k):
            z = s.solve(c * v[:, i - 1])
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
        return [cMax, gMax, cMin, gMin, cMean, gMean, cColVar, gColVar,\
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
            originSolver = splu(csc_matrix(self.g + s * self.c))
            x[:, i] = originSolver.solve(self.b.todense())
            t = solve((rg + s * rc), rb)
            rx[:, i] = v * t

        meanError = abs(x - rx).mean()
        return meanError
            
    def measureMinError(self, totalOrder, maxFreq, step):
        '''
        Find the (frequence, order) of the 2nd expasion point
        to minimum the total error of the 1st and 2nd expansion point.
        The 1st expansion point is always at 0Hz.
        totalOrder: the sum of the order after reducion on point 1 and 2.
        maxFreq: maximum frequency of the 2nd expansion point.
        step: sample step of the 2nd expansion point order.
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
                    error1 = self.errorEval(v1, maxFreq, ERROR_SAMPLE_COUNT)
                    error2 = self.errorEval(v2, maxFreq, ERROR_SAMPLE_COUNT)
                except(np.linalg.LinAlgError):
                    j = j + step
                    continue
                if error1 + error2 <= error:
                    error = error1 + error2
                    result = feature + [i, j, error]
                j = j + step

        return result


class CaseDatabase(object):
    '''
    Database for cases. Training and predicting cases are included.
    Apply access and parser to outer files.
    Attributes: _trainingCaseList, _predictingCaseList, trainingFeatureTable, predictingFeatureTable
    Function: parser, buildCaseData, measureTrainingData, extractPredtingFeature, trainingList, predictingList, 
    saveTrainingFeature, savePredictingFeature, loadTrainingFeature, loadPredictingFeature
    '''

    def __init__(self):
        '''
        Read files from case directory.
        Build _trainingCaseList and _predictingCaseList using buildCaseData function.
        '''
        pass

    def parser(self, fileName):
        '''
        Read and build sparse matrix from file.
        Matrix files in COO format are supported.
        Do not construct matrix since size are unknown.
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
        return self._trainingFeatureTable()
    
    def predictingFeatureTable(self):
        '''
        Access to _predictingFeatureTable.
        '''
        return self._predictingFeatureTable()

    def buildCaseData(self, dataSet = "both", trainingDir = TRAINING_CASE_DIR, predictingDir = PREDICTING_CASE_DIR):
        '''
        Load data from case directory.
        Rebuild _trainingCaseList and/or _predictingCaseList.
        This method do not rebuild _trainingFeatureTable and _predictingFeatureTable.
        dataSet: "training" ---- Rebuild _trainingCaseList from trainingDir
                 "predicting" ---- Rebuild _predictingCaseList from predictingDir
                 "both" ---- rebuild both dir
                 otherwise ---- do not rebuild
        trainingDir: training case directory. TRAINING_CASE_DIR as default.
        predictingDir: predicting case directory. PREDICTING_CASE_DIR as default.
        '''
        pass

    def measureTrainingData(self):
        '''
        Build _trainingFeatureTable from _trainingCaseList.
        This method will delete the existing _trainingFeatureTable.
        May be time-consuming.
        '''
        pass

    def extractPredtingFeature(self):
        '''
        Build _predictingFeatureTable from _predictingCaseList.
        This method will delete the existing _predictingFeatureTable.
        '''

    def saveTrainingFeature(self, fileName = "./training_feature.txt"):
        '''
        Save _trainingFeatureTable to specific file.
        '''
        pass

    def savePredictingFeature(self, fileName = "./predicting_feature.txt"):
        '''
        Save _PredictingFeatureTable to specific file.
        '''
        pass

    def loadTrainingFeature(self, fileName = "./training_feature.txt"):
        '''
        Load _trainingFeatureTable from specific file.
        '''
        pass

    def loadPredictingFeature(self, fileName = "./predicting_feature.txt"):
        '''
        Load _PredictingFeatureTable from specific file.
        '''
        pass



# ---------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------
# Unit Test

if __name__ == "__main__":

    db = CaseDatabase()
    t = CaseData("414", 10, 1)
    [rows, cols, values] = db.parser(PREDICTING_CASE_DIR + "/414/414_B.txt")
    t.setB(coo_matrix((values, (rows, cols)), shape=(10, 1)))
    [rows, cols, values] = db.parser(PREDICTING_CASE_DIR + "/414/414_C.txt")
    t.setC(coo_matrix((values, (rows, cols)), shape=(10, 10)))
    [rows, cols, values] = db.parser(PREDICTING_CASE_DIR + "/414/414_G.txt")
    t.setG(coo_matrix((values, (rows, cols)), shape=(10, 10)))

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
    
    v1 = t.arnoldi(4, 0)
    v2 = t.arnoldi(11, 5e9)

    # print(v)

    # print(t.errorEval(v1, FREQ_LIMIT, ERROR_SAMPLE_COUNT))
    # print(t.errorEval(v2, FREQ_LIMIT, ERROR_SAMPLE_COUNT))
    print(t.errorEval(v1, FREQ_LIMIT, ERROR_SAMPLE_COUNT) + t.errorEval(v2, FREQ_LIMIT, ERROR_SAMPLE_COUNT))

    # print(t.measureMinError(15, FREQ_LIMIT, 1e9))

    











