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
ERRORMAX = 1e10
ERRORSAMPLECOUNT = 1000

class CaseData(object):
    # Data for single case
    # attributes: name, order, bCol, c, g, b
    # function: setB, setC, setG, featureExtract, arnoldi, errorEval, measureMinError

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
        Return -1 if the matrix d = g + s * c is singular
        k: reduction order
        s: expansion point frequency
        '''
        c = self.c.todense()
        g = self.g.todense()
        b = self.b.todense()

        v = np.matrix(np.zeros((self.order, k)))

        d = g + s * c
        if np.linalg.det(d) == 0:
            return -1
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
        error = ERRORMAX

        for i in range(1, totalOrder):
            k1 = totalOrder - i
            k2 = i
            j = step
            while j <= maxFreq:
                s1 = 0
                s2 = j
                v1 = self.arnoldi(k1, s1) 
                v2 = self.arnoldi(k2, s2)
                if(type(v1) == int):
                    error1 = ERRORMAX
                else:
                    error1 = self.errorEval(v1, maxFreq, ERRORSAMPLECOUNT)
                if(type(v2) == int):
                    error2 = ERRORMAX
                else:
                    error2 = self.errorEval(v2, maxFreq, ERRORSAMPLECOUNT)
                if error1 + error2 <= error:
                    error = error1 + error2
                    result = feature + [i, j, error]
                j = j + step

        return result





# ---------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------
# Unit Test

if __name__ == "__main__":
    b = coo_matrix(([-1], ([0], [0])), shape=(10,1))
    c = coo_matrix(([1.000000e-18, 2.388640e-17, 2.388640e-17, 1.319750e-16, 1.145700e-17, 1.145700e-17],\
                    ([1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6])),\
                    shape=(10,10))
    g = np.matrix([[6.171022, 0, 0, 0, -5.080113e+00, 0, 0, 0, 0, -9.090909e-02],\
                  [0, 5.171022, -9.090909e-02, 0, 0, 0, 0, -5.080113e+00, 0, 0],\
                  [0, -9.090909e-02, 8.849164, 0, 0, 0, 0, 0, -8.758255e+00, 0],\
                  [0, 0, 0, 8.767346e+00, 0, 0, -9.090909e-03, 0, 0, -8.758255e+00],\
                  [-5.080113, 0, 0, 0, 5.186578e+00, 0, 0, -1.064646e-01, 0, 0],\
                  [0, 0, 0, 0, 0, 9.090909e-03, 0, 0, -9.090909e-03, 0],\
                  [0, 0, 0, -9.090909e-03, 0, 0, 9.090909e-03, 0, 0, 0],\
                  [0, -5.080113, 0, 0, -1.064646e-01, 0, 0, 5.186578e+00, 0, 0],\
                  [0, 0, -8.758255, 0, 0, -9.090909e-03, 0, 0, 8.767346e+00, 0],\
                  [-9.090909e-02, 0, 0, -8.758255e+00, 0, 0, 0, 0, 0, 8.849164e+00]],)
    
    t = CaseData("414", 10, 1)
    t.setB(b)
    t.setC(c)
    t.setG(g)

    v = t.arnoldi(5, 10e9)

    print(t.errorEval(v, 2e10, 1000))

    print(t.measureMinError(15, 2e10, 1e9))










