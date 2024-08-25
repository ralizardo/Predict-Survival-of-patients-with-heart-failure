#Create the directory in order to load the raw data
import os  
from os import path

import pandas as pd

#script directory
script_path = path.dirname(__file__)

#raw data directory
data_path = path.abspath(path.join(script_path, "..", "..", "data", "raw", "heart_failure_clinical_records_dataset.csv"))


#Reading the raw data
data = pd.read_csv(data_path)

print('csv in variable: data')

