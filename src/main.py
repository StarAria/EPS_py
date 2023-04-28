# ---------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------
# A simple workflow to run the predictor

from case_database import CaseDatabase
from predictor import Predictor

TRAINING_CASE_DIR = "./case/Training"
PREDICTING_CASE_DIR = "./case/Predicting"

db = CaseDatabase()
predicter = Predictor(db)