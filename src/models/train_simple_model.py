from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from scipy.stats import randint
from scipy.stats import uniform
import pandas as pd
import sys
import os  
from os import path
import time
start_time = time.time()

################################################################################
#script directory
script_path = path.dirname(__file__)

#folder of category  split function
stratified_path = path.abspath(path.join(script_path, "..", "data"))#system path
sys.path


# adding ../utils to the system path
sys.path.insert(0, stratified_path)
#################################################################################

#Loading Strat_train_set
import stratified_data as strat

#Loading Simple Pipeline
import simple_pipeline as simple


logistic_reg = make_pipeline( simple.preprocessing, LogisticRegression(class_weight="balanced", max_iter = 1000) )

param_distribs={
    'columntransformer__age_cluster__n_clusters': randint(low=1, high=15)
    , 'columntransformer__age_cluster__gamma':  uniform(0.1, 1)
    , 'columntransformer__time_cluster__n_clusters': randint(low=1, high=14)
    , 'columntransformer__time_cluster__gamma': uniform(0.1, 1)
    , 'columntransformer__ejection_cluster__n_clusters': randint(low=1, high=15)
    , 'columntransformer__ejection_cluster__gamma': uniform(0.1, 1)
    ,'logisticregression__C': uniform(1e-6, 1)
}



x_train = strat.strat_train_set.iloc[:,:-2]

y_train = strat.strat_train_set.iloc[:,-2]



log_search = RandomizedSearchCV( logistic_reg
                                , param_distributions=param_distribs
                                , n_iter=500
                                , cv=15
                                , scoring='recall'
                                , random_state=42)


log_search.fit(x_train, y_train)

print("Training: log_search --- %s seconds ---" % (time.time() - start_time))
