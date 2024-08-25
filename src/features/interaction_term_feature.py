import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler

#directory packages
import sys
import os  
from os import path

#script directory
script_path = path.dirname(__file__)

#folder of category combination function 
cluster_path = path.abspath(path.join(script_path, "..", "utils"))

#system path
sys.path

# adding ../utils to the system path
sys.path.insert(0, cluster_path)

import interaction_term as int 

interaction_pipeline=make_pipeline(
    int.InteractionTermExtractor()
    , FunctionTransformer(np.log,  feature_names_out="one-to-one")
    , StandardScaler() 
)

print('loaded pipeline: interaction_pipeline')

