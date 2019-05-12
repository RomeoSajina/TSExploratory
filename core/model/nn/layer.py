from tensorflow.python.keras.layers import Lambda
from tensorflow.python.keras.layers.core import Dropout
from tensorflow.python.ops import nn
import numpy as np
import scipy.stats as stats


class RandomDropout(Lambda):

    def __init__(self, min_rate=.1, max_rate=.5, seed=np.random.randint(10e6), **kwargs):

        self.max_rate = max_rate
        self.min_rate = min_rate

        def fnc(x):
            rate = self.random_rate()
            return nn.dropout(x*1., rate=rate, seed=seed)

        super(RandomDropout, self).__init__(function=fnc, **kwargs)

    def random_rate(self):

        lower, upper = self.min_rate, self.max_rate
        mu, sigma = 0, 0.3

        X = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
        #N = stats.norm(loc=mu, scale=sigma)

        return X.rvs(1)[0]

"""
class RandomDropout(Dropout):
    
    def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
        super(RandomDropout, self).__init__(rate=rate, noise_shape=noise_shape, seed=seed, **kwargs)
"""