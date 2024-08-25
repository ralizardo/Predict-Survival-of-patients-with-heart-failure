import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler

log_pipeline= make_pipeline(
   FunctionTransformer(np.log,  feature_names_out="one-to-one")
  ,  StandardScaler()
)

print('loaded pipeline: log_pipeline')