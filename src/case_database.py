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
        set B matrix in CaseData.
        The type of B must be coo_matrix
        '''
        if type(b) != coo.coo_matrix:
            print("********Error: type mismatch in B matrix of case " + self.name + " ********\n")
            print("********coo_matrix needed, but " + str(type(b)) + " is given********\n")
            print("********skipping setting B matrix of case " + self.name + " ********\n")
            return
        
        if(b.shape != (self.order, self.bCol)):
            print("********Error: shape mismatch in B matrix of case " + self.name + " ********\n")
            print("********" + str((self.order, self.bCol)) + " needed, but " + str(b.shape) + " is given********\n")
            print("********skipping setting B matrix of case " + self.name + " ********\n")
            return
        
        self.b = b
        return
    
    
