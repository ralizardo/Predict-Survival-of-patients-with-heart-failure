import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

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

#import combination function
import cluster_similarity as cluster

cluster_simil_ = cluster.ClusterSimilarity( n_clusters= 10 , gamma=1., random_state=42)

print('loaded pipeline: cluster_simil_')