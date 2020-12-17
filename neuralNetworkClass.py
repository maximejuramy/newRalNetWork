#######################################################
# STILL TO DO

# Git issue

# Cross-entropy rapidly gets to inf... Tried varying learning rate and batch size : same thing. Cross-entropy def is OK
# => Something wrong in the backprop ? Review the chain rule & how it should be applied in a multiclass case
# Also check feed forward and the way the activations are stored

# Better choice of optimization algo + config (see Stanford courses)

# If Ok, implement more loss & activation functions

# More than one hidden layer ?
#######################################################

import numpy as np
import random

from config import softmax


# Loss & quality evaluation functions
#########################################
def cross_entropy(predicted_scores, true_labels):
    if len(true_labels) == 1:
        raise ValueError(
            "Well you cannot really use Cross-Entropy for"
            + "a regression problem I'm afraid"
        )

    # Only one class equals 1 (all others equal 0) =>
    # so by taking the scalar product and then
    # the log we get the cross-entropy
    return - np.log(true_labels.dot(predicted_scores))


# Main class
##########################

class TwoLayerNeuralNetwork(object):
    def __init__(
        self,
        input_length,
        hidden_layer_length,
        output_length,
        activation_function
    ):
        self.input_length = input_length
        self.hidden_layer_length = hidden_layer_length
        self.output_length = output_length
        self.activation_function = activation_function

        if self.output_length > 1:
            self.output_activation_function = softmax
        else:
            self.output_activation_function = lambda x: x

        # We randomize the weights with a Xavier initialization
        hidden_layer_initialization_std = 1/np.sqrt(self.input_length)
        self.hidden_layer_weights = np.random.normal(
            loc=0.0,
            scale=hidden_layer_initialization_std,
            size=(self.hidden_layer_length, self.input_length)
        )
        self.hidden_layer_bias = np.full((self.hidden_layer_length,), 0.0)

        output_layer_initialization_std = 1/np.sqrt(self.hidden_layer_length)
        self.output_layer_weights = np.random.normal(
            loc=0.0,
            scale=output_layer_initialization_std,
            size=(self.output_length, self.hidden_layer_length)
        )
        self.output_layer_bias = np.full((self.output_length,), 0.0)

    def feed_forward(self, input):
        # The final list will hold a list of lists. Each list matches a layer.
        # Each of these lists holds a list of activations for this layer
        activations = []

        hidden_layer_activations = self.activation_function(
            self.hidden_layer_weights.dot(input) + self.hidden_layer_bias
        )
        activations.append(hidden_layer_activations)

        output_activations = self.output_activation_function(
            self.output_layer_weights.dot(hidden_layer_activations)
            + self.output_layer_bias
        )
        activations.append(output_activations)

        return np.array(activations)

    def predict(self, inputs):
        """
            This method predicts the labels for a given input
        """
        return [self.feed_forward(x)[-1] for x in inputs]

    def back_propagate(self, activations, input, output):
        """
            This functions returns the gradients of the weights and biases of
            the network computed from the given activations (that come from a
            previous feed forward)
        """

        # Handling the output layer
        ############################
        z_output = (
            self.output_layer_weights.dot(activations[0].reshape(
                self.hidden_layer_length, 1)
            ) + self.output_layer_bias.reshape(self.output_length, 1)
        )

        # Gradient of the output layer bias
        nabla_b_output_layer = (
            (2*(activations[-1] - output)).reshape(self.output_length, 1)
            * softmax(z_output)
        )

        # Gradient of the output layer weights
        nabla_w_output_layer = (
            nabla_b_output_layer
        ).dot(activations[0].reshape(self.hidden_layer_length, 1).transpose())

        # Handling the hidden layer
        ############################
        z_hidden = (
            self.hidden_layer_weights.dot(input.reshape(self.input_length, 1))
            + self.hidden_layer_bias.reshape(self.hidden_layer_length, 1)
        )

        nabla_b_hidden_layer = (
            self.output_layer_weights.transpose().dot(nabla_b_output_layer)
            * softmax(z_hidden)
        )

        nabla_w_hidden_layer = (
            nabla_b_hidden_layer
        ).dot(input.reshape(1, self.input_length))

        return (
            nabla_w_hidden_layer, nabla_b_hidden_layer,
            nabla_w_output_layer, nabla_b_output_layer
        )

    def SGD(self, inputs, outputs, epochs, eta, mini_batch_size):
        # Input size
        n = len(inputs)

        # Putting together inputs and outputs before shuffling
        training_data = list(zip(inputs, outputs))

        # Looping on epochs
        for epoch in range(epochs):
            # Create random batches
            random.shuffle(training_data)

            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]

            # Looping on the mini-batches to update the weights and biases
            for batch in mini_batches:
                self.update_mini_batch(batch, eta)

            print(f"Epoch {epoch} concluded")
            training_inputs = [data[0] for data in training_data]
            predicted_scores = self.predict(training_inputs)
            score = 0

            for pred, output in zip(predicted_scores, outputs):
                score += cross_entropy(
                    np.array(pred),
                    np.array(output)
                )

            print(score/n)
            print(
                "The cross-entropy at the end of this epoch is {}\n".format(
                    round(score/len(training_data), 1)
                )
            )

    def update_mini_batch(self, batch, eta):
        """
            This method takes a batch of (input, label)'s and updates (once)
            the weights and biases of the network
        """
        (
            nabla_w_hidden_layer,
            nabla_b_hidden_layer,
            nabla_w_output_layer,
            nabla_b_output_layer
        ) = (
            [], [], [], []
        )
        for (x, y) in batch:
            # Feed forward
            current_activations = self.feed_forward(x)

            # Getting the gradients
            (
                current_nabla_w_hidden_layer, current_nabla_b_hidden_layer,
                current_nabla_w_output_layer, current_nabla_b_output_layer
            ) = self.back_propagate(
                current_activations, np.array(x), np.array(y)
            )

            nabla_w_hidden_layer.append(current_nabla_w_hidden_layer)
            nabla_b_hidden_layer.append(current_nabla_b_hidden_layer)
            nabla_w_output_layer.append(current_nabla_w_output_layer)
            nabla_b_output_layer.append(current_nabla_b_output_layer)

        delta_nabla_w_hidden_layer = sum(nabla_w_hidden_layer)/len(batch)
        delta_nabla_b_hidden_layer = (
            sum(nabla_b_hidden_layer)/len(batch)
        ).reshape(self.hidden_layer_length, 1)
        delta_nabla_w_output_layer = sum(nabla_w_output_layer)/len(batch)
        delta_nabla_b_output_layer = (
            sum(nabla_b_output_layer)/len(batch)
        ).reshape(self.output_length, 1)

        self.hidden_layer_weights -= eta*delta_nabla_w_hidden_layer
        self.hidden_layer_bias -= (
            eta*delta_nabla_b_hidden_layer
        ).reshape(self.hidden_layer_length,)
        self.output_layer_weights -= eta*delta_nabla_w_output_layer
        self.output_layer_bias -= (
            eta*delta_nabla_b_output_layer
        ).reshape(self.output_length,)
