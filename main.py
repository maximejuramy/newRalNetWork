import gzip
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import OneHotEncoder

from neuralNetworkClass import TwoLayerNeuralNetwork
from read_data import get_data
from config import relu

# Reading data
X, y = get_data()

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
