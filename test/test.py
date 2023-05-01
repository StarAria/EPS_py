# ---------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------
# --Project: EPS
# --Date: 2023.04.17
# --
# --File: test.py
# --Note: Test the expansion point predicting method.
# --      For existing 500 cases, choose PREDICTING_CASE_NUM cases randomly for verification, other 450 for training.
# --      Evaluate the method by comparing the predicting expansion point error with mid point error.
# --Designers: Wang Chuyu
# --Writers: Wang Chuyu
# ---------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------

import sys
import random
import matplotlib.pyplot as plt
from numpy import mean

sys.path.append("..")

from EPS.case_database import CaseDatabase
from EPS.predictor import Predictor

FREQ_LIMIT = 50e9
TOTAL_ORDER = 10

PREDICTING_CASE_NUM = 50

# evaluate all the cases
#db = CaseDatabase(trainingCaseDir="../case/Training", predictingCaseDir="../case/Predicting")

# build but do not evaluate. 
db = CaseDatabase(trainingCaseDir="../case/Training", predictingCaseDir="../case/Predicting", build=False)
db.buildCaseData()

# load from file if saved before
db.loadTrainingFeature()

# save the training data if needed.
#db.saveTrainingFeature()



# get the case minimum error table
minErrors = db.trainingFeatureTable()

# randomly partition the dataset
# choose PREDICTING_CASE_NUM cases randomly for verification, others for training
randomLst = sorted(random.sample(range(len(minErrors)), PREDICTING_CASE_NUM))
randomLst.append(len(minErrors))
trainingData = []
predictingData = []
predictingDataPoint = []

j = 0
for i in range(len(minErrors)):
    if(randomLst[j] > i):
        trainingData.append(minErrors[i])
    else:
        predictingData.append(minErrors[i][:-3])
        predictingDataPoint.append(minErrors[i][-3:])
        j = j + 1

# train the model
pd0 = Predictor()
pd0.buildModel(trainingData)
#####pd1 = Predictor()
#####pd1.buildModel(trainingData)
#####pd2 = Predictor()
#####pd2.buildModel(trainingData)
#####pd3 = Predictor()
#####pd3.buildModel(trainingData)
#####pd4 = Predictor()
#####pd4.buildModel(trainingData)

# predict the expansion point
pd0.predict(predictingData)
#####pd1.predict(predictingData)
#####pd2.predict(predictingData)
#####pd3.predict(predictingData)
#####pd4.predict(predictingData)

# measure error with mid point as 2nd expansion point.
midPoints = []
for i in range(PREDICTING_CASE_NUM):
    midPoints.append([pd0.freqPredictingResult[i][0],\
                      FREQ_LIMIT / 2,\
                      int(TOTAL_ORDER / 2)])
midPointError = db.measureCases(midPoints)

# measure error with predict result
ep0 = []
for i in range(PREDICTING_CASE_NUM):
    ep0.append([pd0.freqPredictingResult[i][0],\
                pd0.freqPredictingResult[i][1],\
                pd0.orderPredictingResult[i][1]])
pe0 = db.measureCases(ep0)
#####ep1 = []
#####for i in range(PREDICTING_CASE_NUM):
#####    ep1.append([pd1.freqPredictingResult[i][0],\
#####                pd1.freqPredictingResult[i][1],\
#####                pd1.orderPredictingResult[i][1]])
#####pe1 = db.measureCases(ep1)
#####ep2 = []
#####for i in range(PREDICTING_CASE_NUM):
#####    ep2.append([pd2.freqPredictingResult[i][0],\
#####                pd2.freqPredictingResult[i][1],\
#####                pd2.orderPredictingResult[i][1]])
#####pe2 = db.measureCases(ep2)
#####ep3 = []
#####for i in range(PREDICTING_CASE_NUM):
#####    ep3.append([pd3.freqPredictingResult[i][0],\
#####                pd3.freqPredictingResult[i][1],\
#####                pd3.orderPredictingResult[i][1]])
#####pe3 = db.measureCases(ep3)
#####ep4 = []
#####for i in range(PREDICTING_CASE_NUM):
#####    ep4.append([pd4.freqPredictingResult[i][0],\
#####                pd4.freqPredictingResult[i][1],\
#####                pd4.orderPredictingResult[i][1]])
#####pe4 = db.measureCases(ep4)
#####

# calculate order and frequency deviation
freqDeviation = [(pd0.freqPredictingResult[i][1]-predictingDataPoint[i][1]) \
              for i in range(PREDICTING_CASE_NUM)]
orderDeviation = [(pd0.orderPredictingResult[i][1]-predictingDataPoint[i][0]) \
              for i in range(PREDICTING_CASE_NUM)]
# draw order and frequency deviation figure
plt.figure("Frequency predicting deviation")
plt.stem(freqDeviation)
plt.xlabel("case")
plt.ylabel("frequency deviation")
plt.title("Frequency predicting deviation")
plt.savefig("frequency_predict_deviation")

plt.figure("Order predicting deviation")
plt.stem(orderDeviation)
plt.xlabel("case")
plt.ylabel("order deviation")
plt.title("Order predicting deviation")
plt.savefig("order_predict_deviation")

# calculate error reduction rate
#####predictingError = [[pe0[i][0], min([pe0[i][1], pe1[i][1], pe2[i][1], pe3[i][1], pe4[i][1]])]\
#####                   for i in range(PREDICTING_CASE_NUM)]
#####

predictingError = pe0

errorReductionRate = [((midPointError[i][1] - predictingError[i][1]) / midPointError[i][1])\
                      for i in range(PREDICTING_CASE_NUM)]

meanRateReduction = mean(errorReductionRate)

# plot error and error reduction rate
plt.figure("Error reduction rate")
plt.stem(errorReductionRate)
plt.xlabel("case")
plt.ylabel("error reduction rate")
plt.title("Error reduction rate")
plt.savefig("error_reduction_rate")

plt.figure("Error with predicting point and mid point")
plt.plot(range(PREDICTING_CASE_NUM), [predictingError[i][1] for i in range(PREDICTING_CASE_NUM)],\
      'b', label="error with predicting point", linewidth = 1)
plt.plot(range(PREDICTING_CASE_NUM), [midPointError[i][1] for i in range(PREDICTING_CASE_NUM)],\
      'r', label="error with mid point", linewidth = 1)
plt.yscale('log')
plt.xlabel("case")
plt.ylabel("error")
plt.title("Error with predicting point and mid point")
plt.legend()
plt.savefig("error")

# print result
print("********Result********\n")

print("\tcase\t\terror with mid point\t\terror with predicting point\terror reduction rate")
for i in range(PREDICTING_CASE_NUM):
    print("\t" + predictingError[i][0] + "\t\t" + str(midPointError[i][1]) + "\t\t" +\
          str(predictingError[i][1]) + "\t\t" + str(errorReductionRate[i]))

print("Average error rate reduction: " + str(100*meanRateReduction) + "%\n")

count = len(list(filter(lambda x: x > 0, errorReductionRate)))

print("Improved case count: " + str(count) + " , " + str(len(errorReductionRate)) + " cases in total.\n")