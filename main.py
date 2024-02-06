import numpy as np
import jax.numpy as jnp
import jax
from jaxopt import GradientDescent
from tqdm import tqdm
import matplotlib.pyplot as plt

from conv import Conv
from linear import Linear
from common import *


nh = 2
no = 2
kernel_size = [5, 5]
epoch = 1000
batch_size = 10

conv = Conv(nh, kernel_size)
linear = Linear(nh, no)

conv_w, conv_b = conv.generate_params()
linear_w, linear_b = linear.generate_params()
params = [conv_w, conv_b, linear_w, linear_b]


def predict(params, x):
    conv_w, conv_b, linear_w, linear_b = params
    u_beta = conv.calc_u(conv_w, conv_b, x)
    x_beta = conv.calc_x(u_beta)
    z_beta = conv.calc_z(x_beta)
    u_gamma = linear.calc_u(linear_w, linear_b, z_beta)
    x_gamma = linear.calc_x(u_gamma)
    return x_gamma

@jax.jit
def loss_fn(params, x, y):
    z = predict(params, x)
    tmp = -jnp.sum(y*jnp.log(z+1e-7), axis=1)
    loss = jnp.mean(tmp)
    return loss


train_x, train_y, test_x, test_y = load_dataset()

N = train_x.shape[0]
max_iter = N // batch_size

train_bx = jnp.reshape(train_x, [max_iter, batch_size, 28, 28])
train_by = jnp.reshape(train_y, [max_iter, batch_size, 2])


loss_list = []
loss = loss_fn(params, train_x, train_y)
loss_list.append(loss)

gd = GradientDescent(fun=loss_fn, stepsize=1e-1)

state = gd.init_state(params)

for _ in tqdm(range(epoch)):
    for i in range(max_iter):
        params, state = gd.update(params, state, train_bx[i], train_by[i])
    loss = loss_fn(params, train_x, train_y)
    loss_list.append(loss)

plt.figure()
plt.ylim(0, )
plt.plot(loss_list)
plt.show()
