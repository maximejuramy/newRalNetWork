import gzip
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import OneHotEncoder

from neuralNetworkClass import TwoLayerNeuralNetwork
from config import relu


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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Turning labels into list of 0s and 1s (one hot encoding)
enc = OneHotEncoder(handle_unknown='ignore', sparse=False)

training_labels = enc.fit_transform(y_train.reshape(-1,1))
test_labels = enc.transform(y_test.reshape(-1,1))

# Building network
base_network = TwoLayerNeuralNetwork(
    input_length=28*28, 
    hidden_layer_length=15,
    output_length = 10,
    activation_function=relu
)

print("Initialization output layer weights")
print(base_network.output_layer_weights)

base_network.SGD(
    inputs=X_train,
    outputs=training_labels,
    epochs=20,
    eta=0.5,
    mini_batch_size=10
)

print("\nFinal output layer weights")
print(base_network.output_layer_weights)

print("\nFinal prediction")
print(base_network.feed_forward(X_test[0])[-1])
print(y_test[0])
