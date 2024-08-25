# all pipeline to interacion model
#FINAL we need to store the data in processed folder

from sklearn.compose import ColumnTransformer
# all pipeline for simple model
#directory packages
import sys
import os  
from os import path
####################################################################################################################################################
#script directory
script_path = path.dirname(__file__)


#folder of category combination function 
features_path = path.abspath(path.join(script_path, "..", "features"))



#system path
sys.path

# adding ../utils to the system path
sys.path.insert(0, features_path)
####################################################################################################################################################

#loading individuals pipelines
import interaction_term_feature as intpip
import log_transform_feature as log
import multimodal_distributed_feature as multi
import standard_distributed_feature as std

preprocessing_int =  ColumnTransformer([  ("interaction", intpip.interaction_pipeline, ['age', 'time'])
                                          , ("log", log.log_pipeline, [ "serum_creatinine"])
                                          , ("age_cluster", multi.cluster_simil_ , ["age"]  )
                                          , ("time_cluster", multi.cluster_simil_ , [ 'time']  )
                                          , ("ejection_cluster", multi.cluster_simil_ , ["ejection_fraction"])
                                          , ("default", std.default_pipeline, ["serum_sodium"])
                                      ], remainder='drop')

print('loaded interaction pipeline: preprocessing_int')

