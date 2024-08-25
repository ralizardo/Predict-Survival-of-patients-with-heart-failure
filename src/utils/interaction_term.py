from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures

class InteractionTermExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)

    def fit(self, X, y=None):
        self.poly.fit(X)
        return self

    def transform(self, X):
        poly_features = self.poly.transform(X)
        # El término de interacción se encuentra en la última columna
        interaction_term = poly_features[:, -1]
        return interaction_term.reshape(-1,1)

    def get_feature_names_out(self, names=None):
        interaction_name=self.poly.get_feature_names_out()
        return [interaction_name[2]]
    