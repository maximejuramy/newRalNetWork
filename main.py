from sklearn.model_selection import train_test_split
import numpy as np

from neuralNetworkClass import TwoLayerNeuralNetwork
from read_data import get_data
from config import relu
from metrics import accuracy


"""
# Reading data
X, y = get_data()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
"""
np.random.seed(1987)
X_train_A = 1 + np.random.randn(20, 1)*0.01
X_train_B = np.random.randn(20, 1)*0.01
X_train = np.concatenate([X_train_A, X_train_B])

X_test_A = 1 + np.random.randn(5, 1)*0.01
X_test_B = np.random.randn(5, 1)*0.01
X_test = np.concatenate([X_test_A, X_test_B])

y_train = np.array(['A']*20 + ['B']*20)
y_test = np.array(['A']*5 + ['B']*5)

# Building network
"""
base_network = TwoLayerNeuralNetwork(
    input_length=28*28,
    hidden_layer_length=16,
    output_length=10,
    activation_function=relu
)
"""
base_network = TwoLayerNeuralNetwork(
    input_length=1,
    hidden_layer_length=4,
    output_length=2,
    activation_function=relu
)

base_network.fit(
    inputs=X_train,
    outputs=y_train,
    epochs=50,
    eta=0.01,
    mini_batch_size=2
)

# Predictions and performance of network
predictions = base_network.predict(X_test)

print(f'Accuracy of prediction : {accuracy(y_test, predictions)}%')
