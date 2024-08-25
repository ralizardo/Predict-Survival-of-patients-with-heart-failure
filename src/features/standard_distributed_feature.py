from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler


default_pipeline = make_pipeline(StandardScaler())


print('loaded pipeline: default_pipeline')