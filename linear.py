import numpy as np
import jax.numpy as jnp
from common import softmax

rng = np.random.default_rng(seed=0)

class Linear():

    def __init__(self, nh, no):
        self.nh = nh
        self.no = no
    
    def generate_params(self, c=0.001):
        w = c * rng.normal(0, 1, [self.nh, self.no])
        b = c * rng.normal(0, 1, 1)
        return w, b
    
    def calc_u(self, w, b, x):
        tmp = jnp.matmul(x, w)
        u = tmp + b
        return u

    def calc_x(self, u):
        x = softmax(u, axis=1)
        return x

    def calc_z(self, x):
        z = 1 - x[:,:,0]
        return z
