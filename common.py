import jax.numpy as jnp

def load_dataset():
    train_x = jnp.load(f'./dataset/train_x.npy')
    train_y = jnp.load(f'./dataset/train_y.npy')
    test_x = jnp.load(f'./dataset/test_x.npy')
    test_y = jnp.load(f'./dataset/test_y.npy')
    return train_x, train_y, test_x, test_y

def im2col(image, kernel_size):
    """
        image: [batch_size, 28, 28]
        kernel_size: [5, 5]
    """
    batch_size = image.shape[0]
    image_size = image.shape[1:]
    conved_size = [image_size[0]-kernel_size[0]+1, image_size[1]-kernel_size[1]+1]
    tmp_col = jnp.empty([batch_size, kernel_size[0], kernel_size[1], conved_size[0], conved_size[1]])
    for i in range(kernel_size[0]):
        for j in range(kernel_size[1]):
            tmp = image[:, i:i+conved_size[0], j:j+conved_size[1]]
            tmp_col = tmp_col.at[:,i,j].set(tmp)
    tmp_col = jnp.transpose(tmp_col, [1,2,0,3,4])
    tmp_col = jnp.reshape(tmp_col, [kernel_size[0]*kernel_size[1], batch_size*conved_size[0]*conved_size[1]])
    tmp_col = jnp.transpose(tmp_col)
    tmp_col = jnp.reshape(tmp_col, [batch_size, conved_size[0]*conved_size[1], kernel_size[0]*kernel_size[1]])
    col = jnp.transpose(tmp_col, [0,2,1])
    return col

def softmax(u, axis, t=1.0):
    max = jnp.max(u, axis, keepdims=True)
    bunshi = jnp.exp((u - max) / t)
    bunbo = jnp.sum(bunshi, axis, keepdims=True)
    x = bunshi / bunbo
    return x

def append_u0(u):
    """
        u: [batch_size, N_beta, 576]
    """
    batch_size = u.shape[0]
    N_beta = u.shape[1]
    tmp = jnp.zeros([batch_size, N_beta, 1])
    u = jnp.dstack([tmp, u])
    return u
