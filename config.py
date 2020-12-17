import numpy as np


# Activation functions
##########################
def softmax(output_array):
    exp_x = np.exp(list(output_array)-np.max(list(output_array)))

    return np.array([x/sum(exp_x) for x in exp_x])


def relu(x):
    if isinstance(x, float) or isinstance(x, int):
        return max(0, x)

    try:
        if len(x.shape) != 1:
            raise TypeError("This function expects a one-dimension array")
    except AttributeError:
        raise TypeError(
            "Trying to apply an activation function" +
            "on something that is not an array"
        )

    return np.array([max(0, elem) for elem in list(x)])
