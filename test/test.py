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

FREQ_LIMIT = 500e9
TOTAL_ORDER = 5

PREDICTING_CASE_NUM = 50

# =======================================================================================================================================
## If do not have training_feature.txt, enable the lines below.

## evaluate all the cases
#db = CaseDatabase(trainingCaseDir="../case/Training", predictingCaseDir="../case/Predicting")
#
## save the training data if needed.
#db.saveTrainingFeature()

# =======================================================================================================================================


# =======================================================================================================================================
## If already have training_feature.txt, enable the lines below.

# build but do not evaluate. 
db = CaseDatabase(trainingCaseDir="../case/Training", predictingCaseDir="../case/Predicting", build=False)
db.buildCaseData()

# load from file if saved before
db.loadTrainingFeature("./training_feature.txt")

# =======================================================================================================================================



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

# measure error with left and right quarter point as 2nd expansion point.
leftQuarterPoints = []
for i in range(PREDICTING_CASE_NUM):
    leftQuarterPoints.append([pd0.freqPredictingResult[i][0],\
                              FREQ_LIMIT / 4,\
                              int(TOTAL_ORDER / 2)])
leftQuarterPointError = db.measureCases(leftQuarterPoints)

rightQuarterPoints = []
for i in range(PREDICTING_CASE_NUM):
    rightQuarterPoints.append([pd0.freqPredictingResult[i][0],\
                              FREQ_LIMIT * 3 / 4,\
                              int(TOTAL_ORDER / 2)])
rightQuarterPointError = db.measureCases(rightQuarterPoints)

# measure error with points selected based on first step result as 2nd expansion point.
forthPoints = []
fifthPoints = []
for i in range(PREDICTING_CASE_NUM):
    firstStepMinError = min(midPointError[i][1], leftQuarterPointError[i][1], rightQuarterPointError[i][1])
    firstStepMaxError = max(midPointError[i][1], leftQuarterPointError[i][1], rightQuarterPointError[i][1])
    if(firstStepMinError == midPointError[i][1]):       
        forthPoints.append([pd0.freqPredictingResult[i][0], FREQ_LIMIT * 3 / 8, int(TOTAL_ORDER / 2)])
        fifthPoints.append([pd0.freqPredictingResult[i][0], FREQ_LIMIT * 5 / 8, int(TOTAL_ORDER / 2)])
    elif(firstStepMaxError == midPointError[i][1]):
        forthPoints.append([pd0.freqPredictingResult[i][0], FREQ_LIMIT * 1 / 8, int(TOTAL_ORDER / 2)])
        fifthPoints.append([pd0.freqPredictingResult[i][0], FREQ_LIMIT * 7 / 8, int(TOTAL_ORDER / 2)])
    elif(firstStepMinError == leftQuarterPointError[i][1]):
        forthPoints.append([pd0.freqPredictingResult[i][0], FREQ_LIMIT * 1 / 8, int(TOTAL_ORDER / 2)])
        fifthPoints.append([pd0.freqPredictingResult[i][0], FREQ_LIMIT * 3 / 8, int(TOTAL_ORDER / 2)])
    elif(firstStepMinError == rightQuarterPointError[i][1]):
        forthPoints.append([pd0.freqPredictingResult[i][0], FREQ_LIMIT * 5 / 8, int(TOTAL_ORDER / 2)])
        fifthPoints.append([pd0.freqPredictingResult[i][0], FREQ_LIMIT * 7 / 8, int(TOTAL_ORDER / 2)])

forthPointError = db.measureCases(forthPoints)
fifthPointError = db.measureCases(fifthPoints)

# compare to get second step freq
secondStepFreq = []
for i in range(PREDICTING_CASE_NUM):
    if(forthPointError[i][1] < fifthPointError[i][1]):
        secondStepFreq.append(forthPoints[i][1])
    else:
        secondStepFreq.append(fifthPoints[i][1])
        
#for item in secondStepFreq:
#    print(item)
#    print("\t")
        
# select order
orderSelecionPointError = []
for j in range(1, TOTAL_ORDER):
    orderSelecionPoints = []
    for i in range(PREDICTING_CASE_NUM):
        orderSelecionPoints.append([pd0.freqPredictingResult[i][0], secondStepFreq[i], j])
    orderSelecionPointError.append(db.measureCases(orderSelecionPoints))

humanLikeStrategyError = []
for i in range(PREDICTING_CASE_NUM):
    caseMinError = min([orderSelecionPointError[j][i][1] for j in range(TOTAL_ORDER - 1)])
    #for j in range(TOTAL_ORDER - 1):
    #    if(orderSelecionPointError[j][i][1] == caseMinError):
    #        print(j)
    #        print("\t")
    humanLikeStrategyError.append([pd0.freqPredictingResult[i][0], caseMinError])
    
    

#for i in range(PREDICTING_CASE_NUM):
#    midPoints.append([pd0.freqPredictingResult[i][0],\
#                      FREQ_LIMIT * 7 / 8,\
#                      4])

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

errorIncreaseRate = [((predictingError[i][1] - predictingDataPoint[i][2]) / predictingDataPoint[i][2])\
                      for i in range(PREDICTING_CASE_NUM)]
meanRateIncrease = mean(errorIncreaseRate)

errorReductionRateWithHuman = [((humanLikeStrategyError[i][1] - predictingError[i][1]) / humanLikeStrategyError[i][1])\
                      for i in range(PREDICTING_CASE_NUM)]
meanRateReductionWithHuman = mean(errorReductionRateWithHuman)

#midPointErrorMean = mean([midPointError[i][1] for i in range(PREDICTING_CASE_NUM)])
#predictingErrorMean = mean([predictingError[i][1] for i in range(PREDICTING_CASE_NUM)])
#ErrorMeanReduction = 1 - predictingErrorMean / midPointErrorMean


# plot error reduction rate with human-like strategy
plt.figure("Error reduction rate")
plt.stem(errorReductionRateWithHuman)
plt.xlabel("case")
plt.ylabel("error reduction rate")
plt.title("Error reduction rate")
plt.savefig("error_reduction_rate")

plt.figure("Error with predicting point and human-like strategy")
plt.plot(range(PREDICTING_CASE_NUM), [predictingError[i][1] for i in range(PREDICTING_CASE_NUM)],\
      'r', label="error with predicting point", linewidth = 1.5, marker = 'o', markersize = 2)
plt.plot(range(PREDICTING_CASE_NUM), [humanLikeStrategyError[i][1] for i in range(PREDICTING_CASE_NUM)],\
      'b', label="error with human-like strategy", linewidth = 1, marker = 'o', markersize = 2)
plt.yscale('log')
plt.xlabel("case")
plt.ylabel("error")
plt.title("Error with predicting point and human-like strategy")
plt.legend()
plt.savefig("error_human_like_strategy")

#plot min error, predicting point error and mid point error
#plt.figure("error comparison")
#plt.plot(range(PREDICTING_CASE_NUM), [predictingError[i][1] for i in range(PREDICTING_CASE_NUM)],\
#      'r', label="error with predicting point", linewidth = 1, marker = 'o', markersize = 2)
#plt.plot(range(PREDICTING_CASE_NUM), [midPointError[i][1] for i in range(PREDICTING_CASE_NUM)],\
#      'b', label="error with mid point", linewidth = 1, marker = 'o', markersize = 2)
#plt.plot(range(PREDICTING_CASE_NUM), [predictingDataPoint[i][2] for i in range(PREDICTING_CASE_NUM)],\
#      'g', label="min error", linewidth = 1, marker = 'o', markersize = 2)
#plt.yscale('log')
#plt.xlabel("case")
#plt.ylabel("error")
#plt.title("error comparison")
#plt.legend()
#plt.savefig("error_comparison")

#plot human-like strategy error, predicting point error and mid point error
plt.figure("error comparison")
plt.plot(range(PREDICTING_CASE_NUM), [predictingError[i][1] for i in range(PREDICTING_CASE_NUM)],\
      'r', label="error with predicting point", linewidth = 1, marker = 'o', markersize = 2)
plt.plot(range(PREDICTING_CASE_NUM), [humanLikeStrategyError[i][1] for i in range(PREDICTING_CASE_NUM)],\
      'b', label="error with human-like strategy", linewidth = 1, marker = 'o', markersize = 2)
plt.plot(range(PREDICTING_CASE_NUM), [predictingDataPoint[i][2] for i in range(PREDICTING_CASE_NUM)],\
      'g', label="min error", linewidth = 1, marker = 'o', markersize = 2)
plt.yscale('log')
plt.xlabel("case")
plt.ylabel("error")
plt.title("error comparison")
plt.legend()
plt.savefig("error_comparison")

#plot min error and predicting point error
plt.figure("min error and predicting point error")
plt.plot(range(PREDICTING_CASE_NUM), [predictingError[i][1] for i in range(PREDICTING_CASE_NUM)],\
      'r', label="error with predicting point", linewidth = 1.5, marker = 'o', markersize = 2)
plt.plot(range(PREDICTING_CASE_NUM), [predictingDataPoint[i][2] for i in range(PREDICTING_CASE_NUM)],\
      'g', label="min error", linewidth = 1, marker = 'o', markersize = 2)
plt.yscale('log')
plt.xlabel("case")
plt.ylabel("error")
plt.title("Error with predicting point and min error")
plt.legend()
plt.savefig("error_with_min_error")

# print result
#print("********Compare With mid point********\n")
#
#print("\tcase\t\terror with mid point\t\terror with predicting point\terror reduction rate")
#for i in range(PREDICTING_CASE_NUM):
#    print("\t" + predictingError[i][0] + "\t\t" + str(midPointError[i][1]) + "\t\t" +\
#          str(predictingError[i][1]) + "\t\t" + str(errorReductionRate[i]))
#
#print("Average error rate reduction: " + str(100*meanRateReduction) + "%\n")
#
##print("Error sum reduction: " + str(100*ErrorMeanReduction) + "%\n")
#
#count = len(list(filter(lambda x: x > 0, errorReductionRate)))
#
#print("Improved case count: " + str(count) + " , " + str(len(errorReductionRate)) + " cases in total.\n")

print("********Compare With human-like strategy********\n")

print("\tcase\t\tORError with human-like strategy\tORError with predicting point\tORError reduction rate")
for i in range(PREDICTING_CASE_NUM):
    print("\t" + predictingError[i][0] + "\t\t" + str(humanLikeStrategyError[i][1]) + "\t\t\t" +\
          str(predictingError[i][1]) + "\t\t" + str(errorReductionRateWithHuman[i]))

print("Average Order-Reduction-Error rate reduction: " + str(100*meanRateReductionWithHuman) + "%\n")

#print("Error sum reduction: " + str(100*ErrorMeanReduction) + "%\n")

count = len(list(filter(lambda x: x > 0, errorReductionRateWithHuman)))

print("Improved case count: " + str(count) + " , " + str(len(errorReductionRate)) + " cases in total.\n")

#print("********Compare With Min Error********\n")
#
#print("\tcase\t\tmin error\t\t\terror with predicting point\terror increase rate")
#for i in range(PREDICTING_CASE_NUM):
#    print("\t" + predictingError[i][0] + "\t\t" + str(predictingDataPoint[i][2]) + "\t\t" +\
#          str(predictingError[i][1]) + "\t\t" + str(errorIncreaseRate[i]))
#
#print("Average error rate increase: " + str(100*meanRateIncrease) + "%\n")