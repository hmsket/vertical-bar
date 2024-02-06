import numpy as np
import jax.numpy as jnp
from common import im2col, append_u0, softmax

rng = np.random.default_rng(seed=0)

class Conv():

    def __init__(self, nh, kernel_size):
        self.nh = nh
        self.ks = kernel_size

    def generate_params(self, c=0.001):
        w = c * rng.normal(0, 1, [self.nh, self.ks[0]*self.ks[1]])
        b = c * rng.normal(0, 1, [self.nh, 1])
        return w, b
    
    def calc_u(self, w, b, x):
        col = im2col(x, self.ks)
        tmp = jnp.matmul(w, col)
        tmp = append_u0(tmp)
        u = tmp + b
        return u

    def calc_x(self, u):
        x = softmax(u, axis=2)
        return x

    def calc_z(self, x):
        z = 1 - x[:,:,0]
        return z
