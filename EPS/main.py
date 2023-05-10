# ---------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------
# A simple workflow to run the predictor

from case_database import CaseDatabase
from predictor import Predictor

db = CaseDatabase(trainingCaseDir="../case/Training", predictingCaseDir="../case/Predicting")
db.savePredictingFeature()
db.saveTrainingFeature()
predicter = Predictor(db)