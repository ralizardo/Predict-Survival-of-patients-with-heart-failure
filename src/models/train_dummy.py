#train dummy model

from sklearn.dummy import DummyClassifier
import os  
from os import path
import pandas as pd
import sys
####################################################################################################################################################

#script directory
script_path = path.dirname(__file__)

#folder of category combination function 
stratified_path = path.abspath(path.join(script_path, "..", "data"))

#system path
sys.path

# adding ../utils to the system path
sys.path.insert(0, stratified_path)

#####################################################################################################################################################

import stratified_data as strat

x_train = strat.strat_train_set.iloc[:,:-2]

y_train = strat.strat_train_set.iloc[:,-2]

dummy_clf = DummyClassifier(strategy="stratified")

dummy_clf.fit(x_train, y_train)

Y_hat_dummy= dummy_clf.predict(x_train)


print('fitting: dummy_clf')
