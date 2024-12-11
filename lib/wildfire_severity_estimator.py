from common_imports import *

class WildfireSeverityEstimator(BaseEstimator, RegressorMixin):
    def __init__(self, w1=1, w2=1, w3=1, w4 = 1, w5=1, w6=1, w7=1):
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w4 = w4
        self.w5 = w5
        self.w6 = w6
        self.w7 = w7

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        X = X.clip(lower=0)
        
        # Calculate the severity score using the given weights
        score = (
            self.w1 * np.log(X['AcresBurned'] + 1e-5) +  # Added + 1e-5 avoid log(0)
            self.w2 * (X['Fatalities'] + X['Injuries']) +
            self.w3 * (X['StructuresDamaged'] + 2 * X['StructuresDestroyed']) +
            self.w4 * np.log(X['Duration']+ 1e-5) +  # Added + 1e-5 avoid log(0)
            self.w5 * (X['AirTankers'] + X['Helicopters']) +
            self.w6 * (X['Engines'] + X['CrewsInvolved'] + X['Dozers'] + X['WaterTenders']) +
            self.w7 * X['MajorIncident']
        )
        return score