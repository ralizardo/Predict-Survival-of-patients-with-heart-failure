import pandas as pd
import sys
import os  
from os import path
import time
import joblib
start_time = time.time()


################################################################################
#script directory
script_path = path.dirname(__file__)

#folder of category  split function
models_path = path.abspath(path.join(script_path, "..", "src", "models"))#system path
sys.path


# adding ../utils to the system path
sys.path.insert(0, models_path)
#################################################################################
import train_interaction_model as model

final_model = model.log_search_int.best_estimator_

joblib.dump(final_model, "heart_failure_model.pkl")

print(" heart_failure_model.pkl --- %s seconds ---" % (time.time() - start_time))