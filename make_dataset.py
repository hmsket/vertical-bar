import numpy as np

rng = np.random.default_rng(seed=0)

image_size = [28, 28]

train_x = []
train_y = []
test_x = []
test_y = []


""" make train image """
# exist long bar
N = 300
for i in range(N):
    image = np.zeros(image_size)
    # put long bar
    c = rng.integers(0, image_size[1], 1)
    image[:,c] = 1
    # put mini bar
    n_mini_bar = 3
    for j in range(n_mini_bar):
        mini_bar_size = int(rng.integers(3, 10, 1))
        r = int(rng.integers(0, image_size[0]+1-mini_bar_size, 1))
        c = int(rng.integers(0, image_size[1], 1))
        image[r:r+mini_bar_size, c] = 1
    train_x.append(image)
    train_y.append([0, 1])

# not exist long bar
count = 0
while(True):
    if count == N:
        break
    image = np.zeros(image_size)
    # put mini bar
    n_mini_bar = 3
    for j in range(n_mini_bar):
        mini_bar_size = int(rng.integers(3, 10, 1))
        r = int(rng.integers(0, image_size[0]+1-mini_bar_size, 1))
        c = int(rng.integers(0, image_size[1], 1))
        image[r:r+mini_bar_size, c] = 1
    if image_size[1] in np.sum(image, axis=0):
        continue
    count = count + 1
    train_x.append(image)
    train_y.append([1, 0])


""" make test image """
# exist long bar
N = 100
for i in range(N):
    image = np.zeros(image_size)
    # put long bar
    c = rng.integers(0, image_size[1], 1)
    image[:,c] = 1
    # put mini bar
    n_mini_bar = 3
    for j in range(n_mini_bar):
        mini_bar_size = int(rng.integers(3, 10, 1))
        r = int(rng.integers(0, image_size[0]+1-mini_bar_size, 1))
        c = int(rng.integers(0, image_size[1], 1))
        image[r:r+mini_bar_size, c] = 1
    test_x.append(image)
    test_y.append([0, 1])

# not exist long bar
count = 0
while(True):
    if count == N:
        break
    image = np.zeros(image_size)
    # put mini bar
    n_mini_bar = 3
    for j in range(n_mini_bar):
        mini_bar_size = int(rng.integers(3, 10, 1))
        r = int(rng.integers(0, image_size[0]+1-mini_bar_size, 1))
        c = int(rng.integers(0, image_size[1], 1))
        image[r:r+mini_bar_size, c] = 1
    if image_size[1] in np.sum(image, axis=0):
        continue
    count = count + 1
    test_x.append(image)
    test_y.append([1, 0])


np.save('dataset/train_x', train_x)
np.save('dataset/train_y', train_y)
np.save('dataset/test_x', test_x)
np.save('dataset/test_y', test_y)
