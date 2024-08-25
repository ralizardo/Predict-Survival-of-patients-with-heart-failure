# Here I use the function defined previously with 
# the stratefied strategy by the combination of 
# each category
#THEN eliminate the new field
#separe the data into X and Y variables. 

import pandas as pd
import sys
import os  
from os import path
import load_data
import sklearn
from sklearn.model_selection import train_test_split

#loading data
data = load_data.data

#boolean variables
boolean_columns = ['anaemia',
                    'diabetes',
                    'high_blood_pressure',
                    'sex',
                    'smoking']

#script directory
script_path = path.dirname(__file__)

#folder of category combination function 
combination_path = path.abspath(path.join(script_path, "..", "utils"))

#system path
sys.path

# adding ../utils to the system path
sys.path.insert(0, combination_path)

#import combination function
import category_combination as comb

#Applying category_combination function
data_f = comb.category_combination(data=data, columns = boolean_columns , threshold = 5)

#Stratifying data by combination 
strat_train_set, strat_test_set = train_test_split(
    data_f, test_size=0.2
    , stratify=data_f['combination']
    , random_state=42)

#new directory
train_path = path.abspath(path.join(script_path, "..", "..", "data", "processed", "strat_train_set.csv"))
test_path = path.abspath(path.join(script_path, "..", "..", "data", "processed", "strat_test_set.csv"))

#save train and test set
strat_train_set.to_csv(train_path, index=True)
strat_test_set.to_csv(test_path, index=True)

print('dataframes: strat_train_set, strat_test_set ')


