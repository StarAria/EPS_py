# ---------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------
# --Project: EPS
# --Date: 2023.04.17
# --
# --File: test.py
# --Note: Test the expansion point predicting method.
# --      For existing 500 cases, choose 50 cases randomly for verification, other 450 for training.
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

FREQ_LIMIT = 2e10
TOTAL_ORDER = 15
PROCESS_CNT = 2

PREDICTING_CASE_NUM = 100

# evaluate all the cases
# db = CaseDatabase(trainingCaseDir="../case/Training", predictingCaseDir="../case/Predicting")

# build but do not evaluate. 
db = CaseDatabase(trainingCaseDir="../case/Training", predictingCaseDir="../case/Predicting", build=False)
db.buildCaseData()

# load from file if saved before
db.loadTrainingFeature()

# save the training data if needed.
# db.saveTrainingFeature()

# get the case minimum error table
minErrors = db.trainingFeatureTable()
# get the mid point error table
midPointErrors = db.measureAllCases(FREQ_LIMIT / 2, int(TOTAL_ORDER / 2))

# randomly partition the dataset
# choose PREDICTING_CASE_NUM cases randomly for verification, others for training
randomLst = sorted(random.sample(range(len(minErrors)), PREDICTING_CASE_NUM))
randomLst.append(len(minErrors))
trainingData = []
predictingData = []
predictingDataPoint = []
midPointError = []

j = 0
for i in range(len(minErrors)):
    if(randomLst[j] > i):
        trainingData.append(minErrors[i])
    else:
        predictingData.append(minErrors[i][:-3])
        predictingDataPoint.append(minErrors[i][-3:])
        midPointError.append(midPointErrors[i])
        j = j + 1

# train the model
predictor = Predictor()
predictor.buildModel(trainingData)

# predict the expansion point
predictor.predict(predictingData)

# measure error with predict result
expansionPoints = []
for i in range(PREDICTING_CASE_NUM):
    expansionPoints.append([predictor.freqPredictingResult[i][0],\
                            predictor.freqPredictingResult[i][1],\
                            predictor.orderPredictingResult[i][1]])
predictingError = db.measureCases(expansionPoints)

# calculate order and frequency deviation
freqDeviation = [(predictor.freqPredictingResult[i][1]-predictingDataPoint[i][1]) \
              for i in range(PREDICTING_CASE_NUM)]
orderDeviation = [(predictor.orderPredictingResult[i][1]-predictingDataPoint[i][0]) \
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
errorReductionRate = [((midPointError[i][1] - predictingError[i][1]) / midPointError[i][1])\
                      for i in range(PREDICTING_CASE_NUM)]

meanReduction = mean(errorReductionRate)

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

print("\tcase\t\terror with mid point\t\terror with predicting point\t\terror reduction rate")
for i in range(PREDICTING_CASE_NUM):
    print("\t" + predictingError[i][0] + "\t\t" + str(midPointError[i][1]) + "\t\t" +\
          str(predictingError[i][1]) + "\t\t" + str(errorReductionRate[i]))

print("Average error reduction rate: " + str(100*meanReduction) + "%\n")