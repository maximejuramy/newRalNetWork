import gzip
import numpy as np

def get_data():
    ## Importing data
    image_size = 28
    num_images = 100

    f = gzip.open('data/train-images-idx3-ubyte.gz','r')
    f.read(16)

    buf = f.read(image_size * image_size * num_images)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    X = data.reshape(num_images, image_size*image_size)

    f_y = gzip.open('data/train-labels-idx1-ubyte.gz','r')
    f_y.read(8)

    buf = f_y.read(num_images)
    y = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)

    return X, y