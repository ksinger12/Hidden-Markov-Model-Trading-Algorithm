import numpy as np
from hmmlearn.hmm import GaussianHMM

class HMM():
    """
    iters: number of iterations
    num_regimes: the number of regimes
    model: Hiddnet Markov Model object
    
    Example calls: 
        Initialize: 
        model = HMM(num_regimes = 3, iters = 1000)
        
        Training:
        model.train(train_data)
        
        Prediction:
        regime = model.predict_regime(data_point)
        
        Mean:
        mean = model.means_[regime]
    """
    def __init__(self, num_regimes, iters = 1000):
        self.iters = iters
        self.num_regimes = num_regimes
        self.model = GaussianHMM(n_components = num_regimes, covariance_type = 'diag', n_iter = iters)
        
    def train(self, train_data):
        train_data = list(map(lambda e: [e], train_data))
        self.model.fit(train_data)
    
    def predict_regime(self, data_point):
        data_point = np.array(data_point).reshape(-1, 1)
        current_regime = self.model.predict(data_point)
        return current_regime
    
    def get_regime_mean(self, regime):
        mean = self.model.means_[regime]
        return mean 
        
    def get_transition_probabilities(self, regime):
        return self.model.transmat_[regime, :]
