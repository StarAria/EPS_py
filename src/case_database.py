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
from scipy.sparse.linalg import splu

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
        return -1 if the matrix d = g + s * c is singular
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

    v = t.arnoldi(4, 10e9)

    print(v)











